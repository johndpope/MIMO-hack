import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import clip
import numpy as np
from smplx import SMPLX
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
# from utils import load_video,estimate_depth,detect_and_track_humans,extract_pose,inpaint_scene,compute_masks
from diffusers import UNet3DConditionModel, AutoencoderKL
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.unets.unet_3d_blocks import (
    DownBlock3D,
    UpBlock3D,
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
)
# from diffusers.models.unet_3d_blocks import DownBlock3D, UpBlock3D, CrossAttnDownBlock3D, CrossAttnUpBlock3D
from diffusers import DDIMScheduler,DDPMScheduler
from MimoDataset import MIMODataset
import nvdiffrast.torch as dr

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
        self.ctx = dr.RasterizeGLContext()
        
    def forward(self, vertices, faces, features):
        batch_size, num_vertices, _ = vertices.shape
        device = vertices.device

        # Create perspective projection matrix
        fov = 60
        aspect_ratio = 1.0
        near = 0.1
        far = 100
        proj_mtx = self.perspective_projection(fov, aspect_ratio, near, far)

        # Apply perspective projection
        vertices_proj = self.apply_perspective(vertices, proj_mtx)

        # Prepare vertices for nvdiffrast (clip space)
        vertices_clip = vertices_proj.clone()
        vertices_clip[..., :2] = -vertices_clip[..., :2]
        vertices_clip[..., 2] = 1 - vertices_clip[..., 2]

        # Rasterize
        rast, _ = dr.rasterize(self.ctx, vertices_clip, faces, resolution=[self.image_size, self.image_size])

        # Interpolate features
        feature_maps = dr.interpolate(features, rast, faces)
        
        # Apply simple shading (similar to SoftPhongShader)
        normals = dr.interpolate(self.compute_normals(vertices, faces), rast, faces)
        light_dir = torch.tensor([0, 0, -1], dtype=torch.float32, device=device).expand(batch_size, 3)
        diffuse = torch.sum(normals * light_dir.unsqueeze(1).unsqueeze(1), dim=-1).clamp(min=0)
        shaded_features = feature_maps * diffuse.unsqueeze(-1)
        
        return shaded_features

    def perspective_projection(self, fov, aspect_ratio, near, far):
        fov_rad = np.radians(fov)
        f = 1 / np.tan(fov_rad / 2)
        return torch.tensor([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=torch.float32)

    def apply_perspective(self, vertices, proj_mtx):
        # Convert to homogeneous coordinates
        vertices_hom = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        # Apply projection matrix
        vertices_proj = torch.matmul(vertices_hom, proj_mtx.T.to(vertices.device))
        # Perspective division
        vertices_proj = vertices_proj[..., :3] / vertices_proj[..., 3:]
        return vertices_proj

    def compute_normals(self, vertices, faces):
        v0 = torch.index_select(vertices, 1, faces[:, :, 0])
        v1 = torch.index_select(vertices, 1, faces[:, :, 1])
        v2 = torch.index_select(vertices, 1, faces[:, :, 2])
        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        normals = F.normalize(normals, dim=-1)
        return normals

# pytorch3d
# class DifferentiableRasterizer(nn.Module):
#     def __init__(self, image_size):
#         super(DifferentiableRasterizer, self).__init__()
#         self.image_size = image_size
#         self.raster_settings = RasterizationSettings(
#             image_size=image_size, 
#             blur_radius=0.0, 
#             faces_per_pixel=1
#         )
#         self.renderer = MeshRenderer(
#             rasterizer=MeshRasterizer(raster_settings=self.raster_settings),
#             shader=SoftPhongShader()
#         )
    
#     def forward(self, vertices, faces, features):
#         batch_size, num_vertices, _ = vertices.shape
#         device = vertices.device

#         # Create a batch of meshes
#         meshes = Meshes(verts=vertices, faces=faces.expand(batch_size, -1, -1))
        
#         # Create textures from features
#         textures = TexturesVertex(verts_features=features)
#         meshes.textures = textures

#         # Create dummy cameras (assuming orthographic projection for simplicity)
#         cameras = PerspectiveCameras(device=device, R=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1),
#                                      T=torch.zeros(batch_size, 3), fov=60)

#         # Render the meshes
#         rendered_images = self.renderer(meshes, cameras=cameras)
        
#         # Extract the feature maps (discard alpha channel)
#         feature_maps = rendered_images[..., :features.shape[-1]]
        
#         return feature_maps
class StructuredMotionEncoder(nn.Module):
    def __init__(self, num_vertices, feature_dim, image_size):
        super(StructuredMotionEncoder, self).__init__()
        self.num_vertices = num_vertices
        self.feature_dim = feature_dim
        self.latent_codes = nn.Parameter(torch.randn(num_vertices, feature_dim))
        self.rasterizer = DifferentiableRasterizer(image_size)
        self.smplx = SMPLX('./SMPLX_NEUTRAL.npz',   model_type='smplx',gender='neutral',batch_size=32)
        
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
        print(f"StructuredMotionEncoder input shapes: smpl_params={smpl_params.shape}, camera_params={camera_params.shape}")
        
        batch_size, num_frames, param_dim = smpl_params.shape
        device = smpl_params.device
        
        expanded_codes = self.latent_codes.unsqueeze(0).expand(batch_size * num_frames, -1, -1)
        print(f"Expanded codes shape: {expanded_codes.shape}")
        
        batch_size, num_frames, param_dim = smpl_params.shape
        device = smpl_params.device
        
        # Reshape smpl_params to [batch_size * num_frames, param_dim]
        smpl_params = smpl_params.view(-1, param_dim)
        num_frames = smpl_params.shape[0]
        
        # Split smpl_params into poses and translations
        poses = smpl_params[:, :165]
        trans = smpl_params[:, 165:168]
        
        # Now, split poses into individual components
        global_orient = poses[:, :3]
        body_pose = poses[:, 3:66]
        jaw_pose = poses[:, 66:69]
        leye_pose = poses[:, 69:72]
        reye_pose = poses[:, 72:75]
        left_hand_pose = poses[:, 75:120]
        right_hand_pose = poses[:, 120:165]
        
        
        # Expand betas to match the number of frames
        if betas.shape[0] == 1:
            betas = betas.expand(num_frames, -1)

  # Process frames in batches
        batch_size = 32
        vertices_list = []
        for i in range(0, num_frames, batch_size):
            batch_end = min(i + batch_size, num_frames)
            
            # Extract batch parameters
            batch_betas = betas[i:batch_end]
            batch_global_orient = global_orient[i:batch_end]
            batch_body_pose = body_pose[i:batch_end]
            batch_left_hand_pose = left_hand_pose[i:batch_end]
            batch_right_hand_pose = right_hand_pose[i:batch_end]
            batch_jaw_pose = jaw_pose[i:batch_end]
            batch_leye_pose = leye_pose[i:batch_end]
            batch_reye_pose = reye_pose[i:batch_end]
            batch_trans = trans[i:batch_end]
            
            # Forward pass through SMPL-X model
            smpl_output = self.smpl(
                betas=batch_betas,
                global_orient=batch_global_orient,
                body_pose=batch_body_pose,
                left_hand_pose=batch_left_hand_pose,
                right_hand_pose=batch_right_hand_pose,
                jaw_pose=batch_jaw_pose,
                leye_pose=batch_leye_pose,
                reye_pose=batch_reye_pose,
                transl=batch_trans,
                pose2rot=True
            )
            vertices_list.append(smpl_output.vertices)
        

        vertices = torch.cat(vertices_list, dim=0)
        faces = self.smpl.faces.unsqueeze(0).expand(smpl_params.shape[0], -1, -1)
        print(f"SMPL output shapes: vertices={vertices.shape}, faces={faces.shape}")


        # Rest of the method remains unchanged
        projected_vertices = self.project_to_2d(vertices, camera_params.view(-1, camera_params.shape[-1]))
        print(f"Projected vertices shape: {projected_vertices.shape}")
        
        feature_maps = self.rasterizer(projected_vertices, faces, expanded_codes)
        print(f"Feature maps shape after rasterization: {feature_maps.shape}")
        
        feature_maps = feature_maps.view(batch_size, num_frames, self.feature_dim, self.rasterizer.image_size, self.rasterizer.image_size)
        feature_maps = feature_maps.permute(0, 2, 1, 3, 4)
        print(f"Feature maps shape after reshaping: {feature_maps.shape}")
        
        motion_code = self.encoder(feature_maps)
        print(f"Motion code shape: {motion_code.shape}")
        
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
