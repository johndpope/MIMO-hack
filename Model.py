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
# In DifferentiableRasterizer.forward
import matplotlib.pyplot as plt

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
        print(f"vertex_colors min: {vertex_colors.min()}, max: {vertex_colors.max()}")
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


        projected_vertices_np = vertices_proj.detach().cpu().numpy()[0]
        plt.scatter(projected_vertices_np[:, 0], projected_vertices_np[:, 1], s=1)
        plt.title('Projected Vertices')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.savefig('projected_vertices.png')
        plt.close()
        print("Saved projected vertices visualization.")
        
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

        print(f"feature_maps after interpolation min: {feature_maps.min()}, max: {feature_maps.max()}")


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

    
        print(f"interpolated_normals min: {interpolated_normals.min()}, max: {interpolated_normals.max()}")
        # Compute light direction (you may want to make this configurable)
        light_dir = torch.tensor([0.0, 0.0, 1.0], device=vertices.device).view(1, 1, 1, 3).expand_as(interpolated_normals)

        
        # Compute diffuse shading
        diffuse = torch.sum(interpolated_normals * light_dir, dim=-1, keepdim=True).clamp(min=0)
        print(f"diffuse min: {diffuse.min()}, max: {diffuse.max()}")
        
        # Before applying diffuse shading
        del normals
        del interpolated_normals
        torch.cuda.empty_cache()


        # Apply diffuse shading to feature maps
        # In-place multiplication to save memory
        shaded_features = feature_maps
        shaded_features.mul_(diffuse)
        print(f"Final shaded features shape: {shaded_features.shape}")

    # After computing shaded_features
        shaded_features = shaded_features.permute(0, 3, 1, 2)  # Rearrange to [batch_size, C, H, W]
        image_tensor = shaded_features[0]  # Assuming batch_size is 1

        print(f"feature_maps min: {feature_maps.min()}, max: {feature_maps.max()}")
        print(f"diffuse min: {diffuse.min()}, max: {diffuse.max()}")
        # Save the image
        if torch.isnan(shaded_features).any():
            print("Warning: shaded_features contains NaN values.")

        self.save_rasterized_image(image_tensor)

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

    def save_rasterized_image(self, image_tensor):
        # Convert to NumPy array
        image_np = image_tensor.detach().cpu().numpy()
        print(f"Initial image_np shape: {image_np.shape}")

        # Handle channel dimension
        if image_np.shape[0] >= 3:
            image_np = image_np[:3]  # Take first 3 channels
            image_np = image_np.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        elif image_np.shape[0] == 1:
            image_np = image_np.squeeze(0)  # Grayscale image of shape [H, W]
        else:
            # For other channel sizes, average over channels to get a grayscale image
            image_np = image_np.mean(axis=0)  # Shape: [H, W]
        
        print(f"Processed image_np shape: {image_np.shape}")

        # Check for NaNs and handle them
        if np.isnan(image_np).any():
            print("Warning: image_np contains NaN values. Replacing NaNs with zeros.")
            image_np = np.nan_to_num(image_np, nan=0.0)

        # Normalize and scale
        min_val = image_np.min()
        max_val = image_np.max()
        if max_val - min_val == 0:
            print("Warning: max_val equals min_val. Setting image to gray (128).")
            image_np = np.full_like(image_np, 128, dtype=np.uint8)
        else:
            image_np = (image_np - min_val) / (max_val - min_val)
            image_np = (image_np * 255).astype(np.uint8)

        # Save the image
        from PIL import Image
        try:
            image_pil = Image.fromarray(image_np)
            image_pil.save('rasterized_image.png')
            print("Image saved successfully as 'rasterized_image.png'.")
        except Exception as e:
            print(f"🔥 Error saving image: {e}")
            print(f"image_np shape: {image_np.shape}, dtype: {image_np.dtype}")

        
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
        self.image_size = image_size
        self.latent_codes = nn.Parameter(torch.randn(self.num_vertices, self.feature_dim))
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

        print(f"latent_codes min: {self.latent_codes.min()}, max: {self.latent_codes.max()}")


        # Check the dimensionality of smpl_params
        if smpl_params.dim() == 3:
            batch_size, num_frames, param_dim = smpl_params.shape
        elif smpl_params.dim() == 2:
            batch_size, param_dim = smpl_params.shape
            num_frames = 1
            # Add a singleton dimension to smpl_params and camera_params for consistency
            smpl_params = smpl_params.unsqueeze(1)  # Shape: [batch_size, 1, param_dim]
            camera_params = camera_params.unsqueeze(1)  # Shape: [batch_size, 1, cam_param_dim]
        else:
            raise ValueError(f"Invalid shape for smpl_params: {smpl_params.shape}")

        device = smpl_params.device

        self.smplx = self.smplx.to(device)
        move_to_device(self.smplx, device)

        motion_code_list = []

        for frame_idx in range(num_frames):
            # Extract parameters for the current frame
            frame_smpl_params = smpl_params[:, frame_idx, :]
            frame_camera_params = camera_params[:, frame_idx, :].squeeze(1)

            # Split smpl_params into poses and translations
            poses = frame_smpl_params[:, :165]       # Shape: [batch_size, 165]
            trans = frame_smpl_params[:, 165:168]    # Shape: [batch_size, 3]

            # Now, split poses into individual components
            global_orient = poses[:, :3]             # Indices 0:3
            body_pose = poses[:, 3:66]               # Indices 3:66 (63 parameters)
            jaw_pose = poses[:, 66:69]               # Indices 66:69
            leye_pose = poses[:, 69:72]              # Indices 69:72
            reye_pose = poses[:, 72:75]              # Indices 72:75
            left_hand_pose = poses[:, 75:120]        # Indices 75:120 (45 parameters)
            right_hand_pose = poses[:, 120:165]      # Indices 120:165 (45 parameters)

            # Prepare betas
            frame_betas = betas if betas.shape[0] > 1 else betas.expand(frame_smpl_params.shape[0], -1)

            # Define expression
            num_expression_coeffs = self.smplx.num_expression_coeffs
            expression = torch.zeros(
                [frame_smpl_params.shape[0], num_expression_coeffs],
                dtype=frame_smpl_params.dtype,
                device=frame_smpl_params.device
            )

            # SMPL model forward pass
            smpl_output = self.smplx(
                betas=frame_betas,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                expression=expression,
                transl=trans,  # Use 'trans' instead of 'batch_trans'
                pose2rot=True,
            )
            vertices = smpl_output.vertices  # Shape: [batch_size, num_vertices, 3]



            # Project to 2D
            projected_vertices = self.project_to_2d(vertices, frame_camera_params)

            # Rasterize
            batch_expanded_codes = self.latent_codes.unsqueeze(0).expand(frame_smpl_params.shape[0], -1, -1)
            print(f"batch_expanded_codes shape: {batch_expanded_codes.shape}")
            print(f"batch_expanded_codes min: {batch_expanded_codes.min()}, max: {batch_expanded_codes.max()}")
            feature_map = self.rasterizer(projected_vertices, self.faces_tensor, batch_expanded_codes)

            # Ensure vertex colors match vertices
            assert vertices.shape[1] == batch_expanded_codes.shape[1], "Mismatch in vertices and vertex colors"

            # Reshape feature_map
            feature_map = feature_map.view(frame_smpl_params.shape[0], self.feature_dim, 1, self.image_size, self.image_size)

            # Encode feature_map
            frame_motion_code = self.encoder(feature_map)

            motion_code_list.append(frame_motion_code)

        # Aggregate motion codes
        motion_codes = torch.stack(motion_code_list, dim=2)  # Shape: [batch_size, code_dim, num_frames]

        return motion_codes

        
    def project_to_2d(self, vertices, camera_params):
        print(f"project_to_2d input shapes: vertices={vertices.shape}, camera_params={camera_params.shape}")
        
        batch_size = vertices.shape[0]
        
        # Extract camera parameters
        R = camera_params[:, :9].reshape(batch_size, 3, 3)
        T = camera_params[:, 9:12].reshape(batch_size, 3, 1)
        # Assume intrinsics are provided: fx, fy, cx, cy
        fx = camera_params[:, 12].unsqueeze(-1)  # Shape: [batch_size, 1]
        fy = camera_params[:, 13].unsqueeze(-1)
        cx = camera_params[:, 14].unsqueeze(-1)
        cy = camera_params[:, 15].unsqueeze(-1)
        
        print(f"R shape: {R.shape}, T shape: {T.shape}")
        
        # Apply rotation and translation
        vertices_cam = torch.bmm(vertices, R.transpose(1, 2)) + T.transpose(1, 2)
        
        # Perspective projection
        x = vertices_cam[:, :, 0] / (vertices_cam[:, :, 2] + 1e-7)
        y = vertices_cam[:, :, 1] / (vertices_cam[:, :, 2] + 1e-7)
        
        # Apply camera intrinsics
        u = fx * x + cx  # Shape: [batch_size, num_vertices]
        v = fy * y + cy  # Shape: [batch_size, num_vertices]
        
        # Stack to get projected vertices
        projected_vertices = torch.stack((u, v, vertices_cam[:, :, 2]), dim=-1)  # Shape: [batch_size, num_vertices, 3]
        
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
    def __init__(self,  feature_dim, image_size):
        super().__init__()
        num_vertices = 10475  # Update to match SMPL-X vertices

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
