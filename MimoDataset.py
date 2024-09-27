import os
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import Dict, Any
from decord import VideoReader
from lama import LAMAInpaintingModule, inpaint_scene
from utils import load_video,estimate_depth,detect_and_track_humans,extract_pose,inpaint_scene,compute_masks


class MIMODataset(Dataset):
    def __init__(self, video_folder: str, config_path: str, checkpoint_path: str,
                 sample_size=256, num_frames=200):
        self.dataset = [os.path.join(video_folder, video_path) for video_path in os.listdir(video_folder) if video_path.endswith(("mp4",))]
        random.shuffle(self.dataset)
        self.length = len(self.dataset)

        self.video_folder = video_folder
        self.num_frames = num_frames
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        # Initialize LAMA inpainting module
        self.lama_module = LAMAInpaintingModule(config_path, checkpoint_path)

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
        ])

    def __len__(self) -> int:
        return self.length

    def get_video_frames(self, video_path: str) -> torch.Tensor:
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        
        # Calculate how many times we need to repeat the video
        repeat_times = self.num_frames // video_length + 1
        
        # Create an index list that covers all frames and repeats if necessary
        batch_index = list(range(video_length)) * repeat_times
        batch_index = batch_index[:self.num_frames]
        
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        return pixel_values

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        while True: # add this so additional workers process videos linearly / sequentially without spawning endless threads.
            try:
                video_path = self.dataset[idx]
                name = os.path.basename(video_path)

                # Load video frames
                video = self.get_video_frames(video_path)
                video = self.pixel_transforms(video)

                # Compute depth maps 
                depth_maps = estimate_depth(video)

                # Detect and track humans
                human_masks = detect_and_track_humans(video)

                # Compute masks for spatial decomposition
                human_mask, scene_mask, occlusion_mask = compute_masks(depth_maps, human_masks)

                # Apply masks to get decomposed components
                human_frames = video * human_mask
                scene_frames = video * scene_mask
                occlusion_frames = video * occlusion_mask

                # Inpaint scene frames using LAMA
                inpainting_mask = human_mask | occlusion_mask
                scene_frames = inpaint_scene(scene_frames, inpainting_mask, self.config_path, self.checkpoint_path)

                break
            except Exception as e:
                print(f"Error loading video {self.dataset[idx]}: {e}")
                idx = random.randint(0, self.length-1)

        return {
            'frames': video,
            'video_name': name,
            'human_frames': human_frames,
            'scene_frames': scene_frames,
            'occlusion_frames': occlusion_frames,
            'human_mask': human_mask,
            'scene_mask': scene_mask,
            'occlusion_mask': occlusion_mask
        }