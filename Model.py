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

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


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
        
    def forward(self, vertices, faces, vertex_colors):
        print(f"Faces shape: {faces.shape}, dtype: {faces.dtype}")
        assert faces.dim() == 2 and faces.shape[1] == 3, f"Expected faces to be 2D tensor with 3 columns, got shape {faces.shape}"
        assert faces.dtype == torch.int64, f"Expected faces to be of dtype torch.int64, got {faces.dtype}"
        print("--- DifferentiableRasterizer forward start ---")
        print("Input shapes:")
        print(f"  vertices: {vertices.shape}")
        print(f"  faces: {faces.shape}")
        print(f"  vertex_colors: {vertex_colors.shape}")
        
        batch_size, num_vertices, _ = vertices.shape
        device = vertices.device
        print(f"batch_size: {batch_size}, num_vertices: {num_vertices}, device: {device}")

        faces = faces.to(torch.int32)  # Ensure faces is of int32 type
        faces = faces.contiguous()     # Shape: [num_faces, 3]
        print(f"Faces shape after processing: {faces.shape}")

        vertices = vertices.contiguous()          # Shape: [batch_size, num_vertices, 3]
        vertex_colors = vertex_colors.contiguous()  # Shape: [batch_size, num_vertices, feature_dim]
        print(f"Vertices shape after contiguous: {vertices.shape}")
        print(f"Vertex colors shape after contiguous: {vertex_colors.shape}")

        # Create perspective projection matrix
        fov = 60
        aspect_ratio = 1.0
        near = 0.1
        far = 100
        proj_mtx = self.perspective_projection(fov, aspect_ratio, near, far).to(device)
        print(f"Projection matrix shape: {proj_mtx.shape}")

        # Apply perspective projection
        vertices_proj = self.apply_perspective(vertices, proj_mtx)
        print(f"Projected vertices shape: {vertices_proj.shape}")

        # Prepare vertices for nvdiffrast (clip space)
        vertices_clip = torch.cat([vertices_proj, torch.ones_like(vertices_proj[..., :1])], dim=-1)
        vertices_clip[..., :2] = -vertices_clip[..., :2]
        vertices_clip[..., 2] = vertices_clip[..., 2] * 2 - 1  # Map z from [0, 1] to [-1, 1]
        print(f"Clip space vertices shape: {vertices_clip.shape}")

        print(f"vertices_clip min: {vertices_clip.min()}, max: {vertices_clip.max()}")
        print(f"faces min: {faces.min()}, max: {faces.max()}")

        # Rasterize
        print("Starting rasterization...")
        rast, _ = dr.rasterize(self.ctx, vertices_clip, faces, resolution=[self.image_size, self.image_size])
        print(f"Rasterization output shape: {rast.shape}")

        # Interpolate features
        print("Starting feature interpolation...")
        try:
            feature_maps_tuple = dr.interpolate(vertex_colors, rast, faces)
            if isinstance(feature_maps_tuple, tuple):
                feature_maps, _ = feature_maps_tuple
            else:
                feature_maps = feature_maps_tuple
            print(f"Interpolated feature maps shape: {feature_maps.shape}")
        except Exception as e:
            print(f"Error during feature interpolation: {str(e)}")
            print(f"vertex_colors shape: {vertex_colors.shape}, rast shape: {rast.shape}, faces shape: {faces.shape}")
            raise

        # Compute and interpolate normals
        print("Computing normals...")
        normals = self.compute_normals(vertices, faces)
        print(f"Computed normals shape: {normals.shape}")
        
        print("Interpolating normals...")
        try:
            interpolated_normals = dr.interpolate(normals, rast, faces)
            if isinstance(interpolated_normals, tuple):
                interpolated_normals, _ = interpolated_normals
            print(f"Interpolated normals shape: {interpolated_normals.shape}")
        except Exception as e:
            print(f"Error during normal interpolation: {str(e)}")
            print(f"normals shape: {normals.shape}, rast shape: {rast.shape}, faces shape: {faces.shape}")
            raise

    

        # Compute light direction (you may want to make this configurable)
        light_dir = torch.tensor([0.0, 0.0, 1.0], device=vertices.device).expand_as(interpolated_normals)
        
        
        # Compute diffuse shading
        diffuse = torch.sum(interpolated_normals * light_dir, dim=-1, keepdim=True).clamp(min=0)
        print(f"Diffuse shading shape: {diffuse.shape}")
        
        # Before applying diffuse shading
        del normals
        del interpolated_normals
        torch.cuda.empty_cache()


        # Apply diffuse shading to feature maps
        # In-place multiplication to save memory
        shaded_features = feature_maps
        shaded_features.mul_(diffuse)
        print(f"Final shaded features shape: {shaded_features.shape}")

        return shaded_features
    
    def perspective_projection(self, fov, aspect_ratio, near, far):
        print("--- perspective_projection start ---")
        print(f"Input parameters: fov={fov}, aspect_ratio={aspect_ratio}, near={near}, far={far}")
        fov_rad = np.radians(fov)
        f = 1 / np.tan(fov_rad / 2)
        proj_matrix = torch.tensor([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=torch.float32)
        print(f"Projection matrix:\n{proj_matrix}")
        print("--- perspective_projection end ---")
        return proj_matrix

    def apply_perspective(self, vertices, proj_mtx):
        print("--- apply_perspective start ---")
        print(f"Input vertices shape: {vertices.shape}")
        print(f"Projection matrix shape: {proj_mtx.shape}")
        
        # Convert to homogeneous coordinates
        vertices_hom = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        print(f"Homogeneous vertices shape: {vertices_hom.shape}")
        
        # Apply projection matrix
        vertices_proj = torch.matmul(vertices_hom, proj_mtx.T.to(vertices.device))
        print(f"Projected vertices shape (before division): {vertices_proj.shape}")
        
        # Perspective division
        vertices_proj = vertices_proj[..., :3] / vertices_proj[..., 3:]
        print(f"Final projected vertices shape: {vertices_proj.shape}")
        print(f"Projected vertices min: {vertices_proj.min()}, max: {vertices_proj.max()}")
        print("--- apply_perspective end ---")
        return vertices_proj

    def compute_normals(self, vertices, faces):
        print("--- compute_normals start ---")
        print(f"Input vertices shape: {vertices.shape}")
        print(f"Input faces shape: {faces.shape}")
        
        batch_size, num_vertices, _ = vertices.shape
        num_faces = faces.shape[0]
        print(f"batch_size: {batch_size}, num_faces: {num_faces}")
        
        # Expand faces to match batch size and convert to int64
        faces_expanded = faces.unsqueeze(0).expand(batch_size, -1, -1).to(torch.int64)
        print(f"Expanded faces shape: {faces_expanded.shape}")
        
        # Extract face indices
        f0, f1, f2 = faces_expanded[:, :, 0], faces_expanded[:, :, 1], faces_expanded[:, :, 2]
        print(f"Face indices shapes: f0={f0.shape}, f1={f1.shape}, f2={f2.shape}")
        
        # Gather vertex positions for each face
        v0 = vertices.gather(1, f0.unsqueeze(-1).expand(-1, -1, 3))
        v1 = vertices.gather(1, f1.unsqueeze(-1).expand(-1, -1, 3))
        v2 = vertices.gather(1, f2.unsqueeze(-1).expand(-1, -1, 3))
        print(f"Gathered vertices shapes: v0={v0.shape}, v1={v1.shape}, v2={v2.shape}")
        
        # Compute normals
        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        normals = F.normalize(normals, dim=-1)
        print(f"Computed normals shape: {normals.shape}")
        print(f"Normals min: {normals.min()}, max: {normals.max()}")
        print("--- compute_normals end ---")
        return normals


def move_to_device(module, device):
    for param in module.parameters():
        param.data = param.data.to(device)
    for buffer in module.buffers():
        buffer.data = buffer.data.to(device)
    for child in module.children():
        move_to_device(child, device)
        
class StructuredMotionEncoder(nn.Module):
    def __init__(self, num_vertices, feature_dim, image_size):
        super(StructuredMotionEncoder, self).__init__()
        self.num_vertices = num_vertices
        self.feature_dim = feature_dim
        self.latent_codes = nn.Parameter(torch.randn(num_vertices, feature_dim))
        self.rasterizer = DifferentiableRasterizer(image_size)
        self.smplx = SMPLX('./SMPLX_NEUTRAL.npz',  
                            model_type='smplx',
                            gender='neutral', 
                            use_pca=False,
                        device='cuda' )
        
         # Convert faces to a PyTorch tensor and store it
        self.register_buffer(
            'faces_tensor',
            torch.tensor(self.smplx.faces.astype(np.int64), dtype=torch.long, device='cuda')
        )

        self.encoder = nn.Sequential(
            nn.Conv3d(self.feature_dim, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, 512)
        )

        
    def forward(self, betas, smpl_params, camera_params):
        print(f"StructuredMotionEncoder input shapes: smpl_params={smpl_params.shape}, camera_params={camera_params.shape}")
        print(f"betas device: {betas.device}")
        print(f"smpl_params device: {smpl_params.device}")
        print(f"camera_params device: {camera_params.device}")
        batch_size, num_frames, param_dim = smpl_params.shape
        device = smpl_params.device

        # Ensure SMPLX model is on the correct device
        self.smplx = self.smplx.to(smpl_params.device)
        move_to_device(self.smplx, smpl_params.device)
        
        expanded_codes = self.latent_codes.unsqueeze(0).expand(batch_size * num_frames, -1, -1)
        print(f"Expanded codes shape: {expanded_codes.shape}")
        
        # Reshape smpl_params to [num_frames, param_dim]
        smpl_params = smpl_params.view(-1, param_dim)
        num_frames = smpl_params.shape[0]
        
        # Split smpl_params into poses and translations
        poses = smpl_params[:, :165]       # Shape: [num_frames, 165]
        trans = smpl_params[:, 165:168]    # Shape: [num_frames, 3]
        
        # Now, split poses into individual components
        global_orient = poses[:, :3]             # Indices 0:3
        body_pose = poses[:, 3:66]               # Indices 3:66 (63 parameters)
        jaw_pose = poses[:, 66:69]               # Indices 66:69
        leye_pose = poses[:, 69:72]              # Indices 69:72
        reye_pose = poses[:, 72:75]              # Indices 72:75
        left_hand_pose = poses[:, 75:120]        # Indices 75:120 (45 parameters)
        right_hand_pose = poses[:, 120:165]      # Indices 120:165 (45 parameters)
        
    # Expand betas to match the number of frames
        if betas.shape[0] == 1:
            betas = betas.expand(num_frames, -1)

    # Get number of expression coefficients
        num_expression_coeffs = self.smplx.num_expression_coeffs



  # Process frames in batches
        batch_size = 1
        vertices_list = []
        for i in range(0, num_frames, batch_size):
            batch_end = min(i + batch_size, num_frames)
            batch_size_actual = batch_end - i  # Actual batch size
            
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
            
            # Create expression tensor with the correct batch size
            expression = torch.zeros(
                [batch_size_actual, num_expression_coeffs],
                dtype=batch_betas.dtype,
                device=batch_betas.device
            )
            
            # Forward pass through SMPL-X model
            smpl_output = self.smplx(
                betas=batch_betas,
                global_orient=batch_global_orient,
                body_pose=batch_body_pose,
                left_hand_pose=batch_left_hand_pose,
                right_hand_pose=batch_right_hand_pose,
                jaw_pose=batch_jaw_pose,
                leye_pose=batch_leye_pose,
                reye_pose=batch_reye_pose,
                expression=expression,
                transl=batch_trans,
                pose2rot=True,
                device=device
            )
            vertices_list.append(smpl_output.vertices)
        

        vertices = torch.cat(vertices_list, dim=0)
        faces = self.faces_tensor  # Already on the correct device

        # Flatten camera_params to match the number of frames
        camera_params = camera_params.view(-1, camera_params.shape[-1])  # Shape: [batch_size * num_frames, 12]

        # Prepare expanded_codes
        expanded_codes = self.latent_codes.unsqueeze(0).expand(vertices.shape[0], -1, -1)


        # Process frames in batches
        batch_size_raster = 1  # Adjust as needed
        feature_maps_list = []
        for i in range(0, num_frames, batch_size_raster):
            batch_end = min(i + batch_size_raster, num_frames)
            batch_vertices = vertices[i:batch_end]
            
            print(f"Processing batch {i} to {batch_end}")
            print(f"camera_params shape: {camera_params.shape}")
            print(f"camera_params first few values: {camera_params[:5, :5]}")
            
            # Correctly slice camera_params for the current batch
            if camera_params.dim() == 2:
                batch_camera_params = camera_params[i:batch_end, :]
            elif camera_params.dim() == 3:
                batch_camera_params = camera_params[0, i:batch_end, :]
            elif camera_params.dim() == 4:
                batch_camera_params = camera_params[0, i:batch_end, 0, :]
            else:
                raise ValueError(f"Unexpected camera_params shape: {camera_params.shape}")
            
            print(f"Batch camera params shape: {batch_camera_params.shape}")
            print(f"Batch camera params first few values: {batch_camera_params[:5, :5]}")
            
            batch_projected_vertices = self.project_to_2d(batch_vertices, batch_camera_params)
            batch_expanded_codes = expanded_codes[i:batch_end]

            # Ensure tensors are contiguous
            batch_projected_vertices = batch_projected_vertices.contiguous()
            batch_expanded_codes = batch_expanded_codes.contiguous()

            # Rasterize the batch
            batch_feature_maps = self.rasterizer(batch_projected_vertices, self.faces_tensor, batch_expanded_codes)
            feature_maps_list.append(batch_feature_maps)

        # Concatenate the feature maps from all batches
        feature_maps = torch.cat(feature_maps_list, dim=0)
        
        print(f"Concatenated feature maps shape: {feature_maps.shape}")

        # Reshape feature_maps appropriately
        feature_maps = feature_maps.view(num_frames, self.image_size, self.image_size, self.feature_dim)
        feature_maps = feature_maps.permute(0, 3, 1, 2)  # [num_frames, feature_dim, image_size, image_size]
        feature_maps = feature_maps.unsqueeze(0)  # Add batch dimension
        
        print(f"Reshaped feature maps shape: {feature_maps.shape}")

        # Continue with the rest of your code
        motion_code = self.encoder(feature_maps)

        return motion_code

    def project_to_2d(self, vertices, camera_params):
        print(f"project_to_2d input shapes: vertices={vertices.shape}, camera_params={camera_params.shape}")
        
        batch_size = vertices.shape[0]
        
        # Check if camera_params is empty
        if camera_params.numel() == 0:
            raise ValueError("camera_params is empty. Please ensure camera parameters are properly passed.")
        
        # Ensure tensors are contiguous
        camera_params = camera_params.contiguous()
        vertices = vertices.contiguous()
        
        # Reshape camera parameters
        try:
            R = camera_params[:, :9].reshape(batch_size, 3, 3)
            T = camera_params[:, 9:12].reshape(batch_size, 3, 1)
        except RuntimeError as e:
            print(f"Error reshaping camera parameters: {e}")
            print(f"camera_params shape: {camera_params.shape}")
            raise
        
        print(f"R shape: {R.shape}, T shape: {T.shape}")
        
        # Apply rotation and translation
        projected_vertices = torch.bmm(vertices, R.transpose(1, 2)) + T.transpose(1, 2)
        
        # Perspective division
        projected_vertices = projected_vertices / (projected_vertices[:, :, 2:3] + 1e-7)  # Added small epsilon to avoid division by zero
        
        print(f"projected_vertices shape: {projected_vertices.shape}")
        
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
