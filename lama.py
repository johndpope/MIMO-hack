import torch
import torch.nn as nn
import yaml
import os
from omegaconf import OmegaConf
from typing import List, Tuple

# Assuming you have cloned the LAMA repository and it's in your Python path
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.data import pad_img_to_modulo

class LAMAInpaintingModule(nn.Module):
    def __init__(self, config_path: str, checkpoint_path: str):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        with open(config_path, 'r') as f:
            config = OmegaConf.create(yaml.safe_load(f))
        
        # Load model
        self.model = load_checkpoint(config, checkpoint_path, strict=False, map_location=self.device)
        self.model.eval()
    
    @torch.no_grad()
    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are in the correct format
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
        
        # Move to device
        image = move_to_device(image, self.device)
        mask = move_to_device(mask, self.device)
        
        # Prepare input
        batch = {}
        batch['image'] = image
        batch['mask'] = mask
        
        # Pad image and mask to be divisible by 8
        batch = pad_img_to_modulo(batch, 8)
        
        # Inpaint
        batch['inpainted'] = self.model(batch['image'], batch['mask'])
        
        # Unpad the result
        cur_res = batch['inpainted']
        orig_height, orig_width = image.shape[2:]
        cur_res = cur_res[:, :, :orig_height, :orig_width]
        
        return cur_res.squeeze(0)

def inpaint_scene(scene_frames: torch.Tensor, masks: torch.Tensor, config_path: str, checkpoint_path: str) -> torch.Tensor:
    """
    Use LAMA for advanced inpainting of scene frames.

    Args:
        scene_frames (torch.Tensor): Scene frames tensor of shape [T, C, H, W]
        masks (torch.Tensor): Binary masks tensor of shape [T, 1, H, W], where 1 indicates areas to be inpainted
        config_path (str): Path to the LAMA model config file
        checkpoint_path (str): Path to the LAMA model checkpoint

    Returns:
        torch.Tensor: Inpainted scene frames of shape [T, C, H, W]
    """
    inpainting_module = LAMAInpaintingModule(config_path, checkpoint_path)
    
    inpainted_frames: List[torch.Tensor] = []
    
    for frame, mask in zip(scene_frames, masks):
        inpainted = inpainting_module(frame, mask)
        inpainted_frames.append(inpainted.cpu())
    
    return torch.stack(inpainted_frames)