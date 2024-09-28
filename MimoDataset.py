import os
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import Dict, Any
from decord import VideoReader
from utils import load_video, estimate_depth_sapien, compute_masks, inpaint_scene
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np

class MIMODataset(Dataset):
    def __init__(self, video_folder: str, sam_config: str, sam_checkpoint: str, 
                 lama_config: str, lama_checkpoint: str, sample_size=256, num_frames=200):
        self.dataset = [os.path.join(video_folder, video_path) for video_path in os.listdir(video_folder) if video_path.endswith(("mp4",))]
        random.shuffle(self.dataset)
        self.length = len(self.dataset)
        
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.lama_config = lama_config
        self.lama_checkpoint = lama_checkpoint
        
        # Initialize SAM2 model
        self.sam_model = build_sam2(config_file=sam_config, ckpt_path=sam_checkpoint)
        self.sam_predictor = SAM2ImagePredictor(self.sam_model)

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
        
        repeat_times = self.num_frames // video_length + 1
        batch_index = list(range(video_length)) * repeat_times
        batch_index = batch_index[:self.num_frames]
        
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader
        
        return pixel_values

    def detect_and_track_humans(self, frames):
        human_masks = []
        prev_boxes = None
        
        for frame in frames:
            frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype('uint8')
            self.sam_predictor.set_image(frame_np)
            
            h, w = frame_np.shape[:2]
            input_point = np.array([[w // 2, h // 2]])
            input_label = np.array([1])
            
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            best_mask = masks[np.argmax(scores)]
            mask_tensor = torch.from_numpy(best_mask).float()
            
            if prev_boxes is not None:
                current_box = self.mask_to_box(mask_tensor)
                iou = self.box_iou(current_box.unsqueeze(0), prev_boxes)
                if iou.max() > 0.5:
                    matched_index = iou.argmax()
                    mask_tensor = human_masks[-1][matched_index]
            
            human_masks.append(mask_tensor.unsqueeze(0))
            prev_boxes = self.mask_to_box(mask_tensor).unsqueeze(0)
        
        return torch.cat(human_masks, dim=0)

    def mask_to_box(self, mask):
        y, x = torch.where(mask > 0.5)
        return torch.tensor([x.min(), y.min(), x.max(), y.max()])

    def box_iou(self, box1, box2):
        x1, y1, x2, y2 = box1.unbind(-1)
        x1_, y1_, x2_, y2_ = box2.unbind(-1)
        
        w1, h1 = x2 - x1, y2 - y1
        w2, h2 = x2_ - x1_, y2_ - y1_
        
        left = torch.max(x1, x1_)
        right = torch.min(x2, x2_)
        top = torch.max(y1, y1_)
        bottom = torch.min(y2, y2_)
        
        intersection = (right - left).clamp(min=0) * (bottom - top).clamp(min=0)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / (union + 1e-6)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        while True:
            try:
                video_path = self.dataset[idx]
                name = os.path.basename(video_path)
                
                # Load video frames
                video = self.get_video_frames(video_path)
                video = self.pixel_transforms(video)

                # Compute depth maps 
                depth_maps = estimate_depth_sapien(video)

                # Detect and track humans using SAM2
                human_masks = self.detect_and_track_humans(video)

                # Compute masks for spatial decomposition
                human_mask, scene_mask, occlusion_mask = compute_masks(depth_maps, human_masks, self.sam_predictor)

                # Apply masks to get decomposed components
                human_frames = video * human_mask.unsqueeze(1)
                scene_frames = video * scene_mask.unsqueeze(1)
                occlusion_frames = video * occlusion_mask.unsqueeze(1)

                # Inpaint scene frames
                inpainting_mask = human_mask | occlusion_mask
                scene_frames = inpaint_scene(scene_frames, inpainting_mask, self.lama_config, self.lama_checkpoint)

                break
            except Exception as e:
                print(f"Error loading video {self.dataset[idx]}: {e}")
                idx = random.randint(0, self.length-1)
        
        return {
            'original_frames': video,
            'video_name': name,
            'human_frames': human_frames,
            'scene_frames': scene_frames,
            'occlusion_frames': occlusion_frames,
            'human_mask': human_mask,
            'scene_mask': scene_mask,
            'occlusion_mask': occlusion_mask
        }