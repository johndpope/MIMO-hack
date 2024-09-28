import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import clip
import numpy as np
from smplx import SMPL
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from utils import load_video,estimate_depth,detect_and_track_humans,extract_pose,inpaint_scene,compute_masks
from diffusers import UNet3DConditionModel, AutoencoderKL
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.unet_3d_blocks import DownBlock3D, UpBlock3D, CrossAttnDownBlock3D, CrossAttnUpBlock3D
from diffusers import DDIMScheduler,DDPMScheduler
from MimoDataset import MIMODataset


class TemporalAttentionLayer(nn.Module):
    def __init__(self, channels, num_head_channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.attention = BasicTransformerBlock(channels, num_head_channels, attention_type="temporal")

    def forward(self, x):
        batch, channel, frames, height, width = x.shape
        residual = x
        x = self.norm(x)
        x = x.view(batch, channel, frames, -1).permute(0, 2, 1, 3).contiguous()
        x = x.view(batch * frames, channel, -1).transpose(1, 2)
        x = self.attention(x)
        x = x.transpose(1, 2).view(batch, frames, channel, -1).permute(0, 2, 1, 3)
        x = x.view(batch, channel, frames, height, width)
        return x + residual

class TemporalUNet3DConditionModel(UNet3DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add temporal attention layers
        for block in self.down_blocks + self.up_blocks:
            if isinstance(block, (CrossAttnDownBlock3D, CrossAttnUpBlock3D)):
                for layer in block.attentions:
                    layer.transformer_blocks.append(TemporalAttentionLayer(layer.channels, layer.num_head_channels))


class DifferentiableRasterizer(nn.Module):
    def __init__(self, image_size):
        super(DifferentiableRasterizer, self).__init__()
        self.image_size = image_size
        self.raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=self.raster_settings),
            shader=SoftPhongShader()
        )
    
    def forward(self, vertices, faces, features):
        batch_size, num_vertices, _ = vertices.shape
        device = vertices.device

        # Create a batch of meshes
        meshes = Meshes(verts=vertices, faces=faces.expand(batch_size, -1, -1))
        
        # Create textures from features
        textures = TexturesVertex(verts_features=features)
        meshes.textures = textures

        # Create dummy cameras (assuming orthographic projection for simplicity)
        cameras = PerspectiveCameras(device=device, R=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1),
                                     T=torch.zeros(batch_size, 3), fov=60)

        # Render the meshes
        rendered_images = self.renderer(meshes, cameras=cameras)
        
        # Extract the feature maps (discard alpha channel)
        feature_maps = rendered_images[..., :features.shape[-1]]
        
        return feature_maps

class StructuredMotionEncoder(nn.Module):
    def __init__(self, num_vertices, feature_dim, image_size):
        super(StructuredMotionEncoder, self).__init__()
        self.num_vertices = num_vertices
        self.feature_dim = feature_dim
        self.latent_codes = nn.Parameter(torch.randn(num_vertices, feature_dim))
        self.rasterizer = DifferentiableRasterizer(image_size)
        self.smpl = SMPL('./SMPLX_NEUTRAL.npz', batch_size=1)
        
        self.encoder = nn.Sequential(
            nn.Conv3d(feature_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, 512)
        )
    
    def forward(self, smpl_params, camera_params):
        batch_size, num_frames, _ = smpl_params.shape
        device = smpl_params.device
        
        # Expand latent codes for batch processing
        expanded_codes = self.latent_codes.unsqueeze(0).expand(batch_size * num_frames, -1, -1)
        
        # Process SMPL parameters
        smpl_params = smpl_params.view(-1, smpl_params.shape[-1])
        smpl_output = self.smpl(
            betas=smpl_params[:, :10],
            body_pose=smpl_params[:, 10:72],
            global_orient=smpl_params[:, 72:75],
            pose2rot=False
        )
        vertices = smpl_output.vertices
        faces = self.smpl.faces.unsqueeze(0).expand(batch_size * num_frames, -1, -1)

        # Project to 2D using camera parameters
        projected_vertices = self.project_to_2d(vertices, camera_params.view(-1, camera_params.shape[-1]))
        
        # Rasterize to create 2D feature maps
        feature_maps = self.rasterizer(projected_vertices, faces, expanded_codes)
        
        # Reshape feature maps to include temporal dimension
        feature_maps = feature_maps.view(batch_size, num_frames, self.feature_dim, self.rasterizer.image_size, self.rasterizer.image_size)
        feature_maps = feature_maps.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        
        # Encode feature maps to motion code
        motion_code = self.encoder(feature_maps)
        
        return motion_code
    
    def project_to_2d(self, vertices, camera_params):
        # Simplified projection, assuming camera_params contains rotation and translation
        R = camera_params[:, :9].view(-1, 3, 3)
        T = camera_params[:, 9:12]
        
        # Apply rotation and translation
        projected_vertices = torch.bmm(vertices, R.transpose(1, 2)) + T.unsqueeze(1)
        
        # Perspective division
        projected_vertices = projected_vertices / projected_vertices[:, :, 2:]
        
        return projected_vertices

class CanonicalIdentityEncoder(nn.Module):
    def __init__(self, clip_model):
        super(CanonicalIdentityEncoder, self).__init__()
        self.clip_encoder = clip_model.visual
        
        # Reference-net architecture
        self.reference_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512)
        )
    
    def forward(self, canonical_image):
        global_feature = self.clip_encoder(canonical_image)
        local_feature = self.reference_net(canonical_image)
        return torch.cat([global_feature, local_feature], dim=1)

class SceneOcclusionEncoder(nn.Module):
    def __init__(self):
        super(SceneOcclusionEncoder, self).__init__()
        # Use a pre-trained VAE encoder
        self.vae_encoder = models.resnet50(pretrained=True)
        self.vae_encoder.fc = nn.Identity()  # Remove the final FC layer
        
        # Add temporal convolution layers
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(2048, 1024, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(1024, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU()
        )
    
    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(-1, channels, height, width)
        features = self.vae_encoder(x)
        features = features.view(batch_size, num_frames, -1).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        temporal_features = self.temporal_conv(features)
        return temporal_features.squeeze(-1).squeeze(-1).permute(0, 2, 1)


class DiffusionDecoder(nn.Module):
    def __init__(self, condition_dim):
        super().__init__()
        self.unet = TemporalUNet3DConditionModel(
            sample_size=(16, 64, 64),  # Adjust based on your video dimensions
            in_channels=4,  # 4 channels for latent space
            out_channels=4,
            down_block_types=(
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ),
            up_block_types=(
                "UpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
            ),
            block_out_channels=(320, 640, 1280, 1280),
            layers_per_block=2,
            cross_attention_dim=condition_dim,
            attention_head_dim=8,
        )
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        
    def forward(self, x, timesteps, condition):
        return self.unet(x, timesteps, encoder_hidden_states=condition).sample
    
    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample()
    
    def decode(self, x):
        return self.vae.decode(x).sample
    
class MIMOModel(nn.Module):
    def __init__(self, num_vertices, feature_dim, image_size):
        super().__init__()
        self.motion_encoder = StructuredMotionEncoder(num_vertices, feature_dim, image_size)
        
        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.identity_encoder = CanonicalIdentityEncoder(clip_model)
        
        self.scene_occlusion_encoder = SceneOcclusionEncoder()
        
        condition_dim = 512 + 512 + 1024  # identity + motion + scene_occlusion
        self.decoder = DiffusionDecoder(condition_dim)
        
        self.condition_proj = nn.Linear(condition_dim, condition_dim)
    
    def forward(self, noisy_latents, timesteps, identity_image, smpl_params, camera_params, scene_frames, occlusion_frames):
        identity_code = self.identity_encoder(identity_image)
        motion_code = self.motion_encoder(smpl_params, camera_params)
        scene_code = self.scene_occlusion_encoder(scene_frames)
        occlusion_code = self.scene_occlusion_encoder(occlusion_frames)
        scene_occlusion_code = torch.cat([scene_code, occlusion_code], dim=-1)
        
        # Combine all condition codes
        condition = torch.cat([identity_code, motion_code, scene_occlusion_code], dim=-1)
        condition = self.condition_proj(condition)
        
        noise_pred = self.decoder(noisy_latents, timesteps, condition)
        return noise_pred
