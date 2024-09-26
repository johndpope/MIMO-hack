import torch
import cv2
import numpy as np
from torchvision.transforms import functional as F

def prepare_dataset_item(video_path, reference_image_path):
    # Load video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # Load reference image
    identity_image = cv2.imread(reference_image_path)
    identity_image = cv2.cvtColor(identity_image, cv2.COLOR_BGR2RGB)
    identity_image = F.to_tensor(identity_image)
    
    # Prepare frames
    frames = [F.to_tensor(frame) for frame in frames]
    
    # Simplified SMPL parameters (placeholder)
    smpl_params = torch.zeros((len(frames), 72 + 10))  # 72 for pose, 10 for shape
    
    # Simplified camera parameters
    camera_params = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 10.0]
    ], dtype=torch.float32).repeat(len(frames), 1)
    
    # Simplified scene frames (blurred original frames)
    scene_frames = []
    for frame in frames:
        blurred = F.gaussian_blur(frame, kernel_size=[21, 21])
        scene_frames.append(blurred)
    scene_frames = torch.stack(scene_frames)
    
    # Placeholder occlusion frames (blank frames)
    occlusion_frames = torch.zeros_like(scene_frames)
    
    return {
        'identity_image': identity_image,
        'smpl_params': smpl_params,
        'camera_params': camera_params,
        'scene_frames': scene_frames,
        'occlusion_frames': occlusion_frames,
        'frames': torch.stack(frames)  # Original video frames
    }

# Usage
dataset_item = prepare_dataset_item('path/to/video.mp4', 'path/to/reference_image.jpg')