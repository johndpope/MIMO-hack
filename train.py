import torch

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler,DDPMScheduler
from MimoDataset import MIMODataset
from Model import MIMOModel


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)




noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


def train(model, dataloader, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            
            x_0 = batch['frames'].to(device)
            identity_image = batch['identity_image'].to(device)
            smpl_params = batch['smpl_params'].to(device)
            camera_params = batch['camera_params'].to(device)
            scene_frames = batch['scene_frames'].to(device)
            occlusion_frames = batch['occlusion_frames'].to(device)
            
            # Encode frames to latent space
            with torch.no_grad():
                x_0_latent = model.decoder.encode(x_0)
            
            # Sample noise and timesteps
            noise = torch.randn_like(x_0_latent)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x_0_latent.shape[0],), device=device).long()
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(x_0_latent, noise, timesteps)
            
            # Predict noise
            noise_pred = model(noisy_latents, timesteps, identity_image, smpl_params, camera_params, scene_frames, occlusion_frames)
            
            # Compute loss
            loss = F.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


# Hyperparameters
num_vertices = 6890  # SMPL model has 6890 vertices
feature_dim = 64
image_size = 64
timesteps = 1000
batch_size = 4
num_epochs = 50
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare beta schedule
betas = linear_beta_schedule(timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# Instantiate the model
model = MIMOModel(num_vertices, feature_dim, image_size).to(device)

# Prepare dataset and dataloader
# Assume we have the necessary data

video_paths = ["path/to/video1.mp4", "path/to/video2.mp4", ...]
identity_image_paths = ["path/to/identity1.jpg", "path/to/identity2.jpg", ...]
smpl_params = [...]  # List of SMPL parameters for each video
camera_params = [...]  # List of camera parameters for each video
config_path = "path/to/lama/config.yaml"
checkpoint_path = "path/to/lama/checkpoint.pth"


dataset = MIMODataset(video_paths, identity_image_paths, smpl_params, camera_params, config_path, checkpoint_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train(model, dataloader, optimizer, num_epochs, device)

# Save the trained model
torch.save(model.state_dict(), 'mimo_model.pth')

# Example of inference
# Load the trained model
model.load_state_dict(torch.load('mimo_model.pth'))
model.eval()

# Prepare input data for inference
identity_image = ...  # Load and preprocess identity image
smpl_params = ...  # Prepare SMPL parameters for desired motion
camera_params = ...  # Prepare camera parameters
scene_frames = ...  # Prepare scene frames
occlusion_frames = ...  # Prepare occlusion frames

# Generate video
# generated_video = inference(model, identity_image, smpl_params, camera_params, scene_frames, occlusion_frames, device, timesteps)
