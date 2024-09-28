import torch
import json
import numpy as np
from Model import StructuredMotionEncoder

def load_camera_params(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    camera_params = []
    for camera in data.values():
        R = torch.tensor(camera['R'], dtype=torch.float32).flatten()
        T = torch.tensor(camera['T'], dtype=torch.float32)
        camera_params.append(torch.cat([R, T]))
    
    return torch.stack(camera_params)

def load_smpl_params(npy_file):
    return torch.tensor(np.load(npy_file), dtype=torch.float32)


def load_smplx_params(npz_file):
    data = np.load(npz_file)
    
    # Extract relevant parameters
    betas = torch.tensor(data['betas'], dtype=torch.float32)
    global_orient = torch.tensor(data['global_orient'], dtype=torch.float32)
    body_pose = torch.tensor(data['body_pose'], dtype=torch.float32)
    
    # Combine parameters
    smplx_params = torch.cat([betas, global_orient, body_pose], dim=-1)
    
    return smplx_params


def test_motion_encoder():
    # Initialize the StructuredMotionEncoder
    num_vertices = 6890  # SMPL model typically has 6890 vertices
    feature_dim = 64  # Adjust as needed
    image_size = 256  # Adjust based on your rendered image size
    motion_encoder = StructuredMotionEncoder(num_vertices, feature_dim, image_size)

    # Load camera parameters
    camera_params = load_camera_params('./data/001/camera.json')

    # Load SMPLX parameters
    smplx_params = load_smplx_params('./data/001/smplx.npz')

      # Ensure the data is in the correct shape
    num_frames = smplx_params.shape[0]
    num_cameras = camera_params.shape[0]
    print("num_cameras:",num_cameras)
    # Repeat camera params for each frame
    camera_params = camera_params.unsqueeze(0).repeat(num_frames, 1, 1)

    # Add batch dimension
    smplx_params = smplx_params.unsqueeze(0)
    camera_params = camera_params.unsqueeze(0)

    # Move tensors to the same device as the model
    device = next(motion_encoder.parameters()).device
    smplx_params = smplx_params.to(device)
    camera_params = camera_params.to(device)

    # Forward pass
    with torch.no_grad():
        motion_code = motion_encoder(smplx_params, camera_params)

    print(f"Motion code shape: {motion_code.shape}")
    print(f"Motion code sample: {motion_code[0, :10]}")  # Print first 10 values of the motion code


if __name__ == "__main__":
    test_motion_encoder()