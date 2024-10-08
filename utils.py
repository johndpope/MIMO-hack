import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision.io import read_video
from torchvision.transforms import Resize
# from midas.model_loader import load_model as load_midas_model
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from smplx import SMPL
# from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
import os 
from omegaconf import OmegaConf
from typing import List, Tuple
from skimage.measure import label
import numpy as np
import torch
import numpy as np
from PIL import Image
from torchvision.io import read_video
from torchvision.transforms import Resize
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_smplx_pose(smplx_model, betas, global_orient, body_pose, transl=None, save_path='smplx_pose',camera_id='0'):
    # Ensure inputs are on the correct device
    device = next(smplx_model.parameters()).device
    betas = betas.to(device)
    global_orient = global_orient.to(device)
    body_pose = body_pose.to(device)
    if transl is not None:
        transl = transl.to(device)

    # Forward pass through SMPL-X model
    output = smplx_model(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        return_verts=True
    )

    # Get vertices and faces
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = smplx_model.faces

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh
    mesh = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                           triangles=faces, shade=True, color='cyan', alpha=0.8)

    # Set equal aspect ratio
    ax.set_box_aspect((np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1]), np.ptp(vertices[:, 2])))

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title('SMPL-X Pose Visualization')
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path+camera_id+'.png')
    plt.close(fig)  # Close the figure to free up memory

    print(f"SMPL-X pose visualization saved to {save_path}")


def load_video(video_path):
    """Load video frames as tensors."""
    frames, _, _ = read_video(video_path)
    frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]
    return frames


def load_sapiens_model(model_size="1b", device="cuda"):

    CHECKPOINTS = {
        "0.3b": "/media/oem/12TB/sapiens/pretrain/sapiens_lite_host/torchscript/depth/checkpoints/sapiens_0.3b/sapiens_0.3b_render_people_epoch_100_torchscript.pt2",
        "0.6b": "/media/oem/12TB/sapiens/pretrain/sapiens_lite_host/torchscript/depth/checkpoints/sapiens_0.6b/sapiens_0.6b_render_people_epoch_70_torchscript.pt2",
        "1b": "/media/oem/12TB/sapiens/pretrain/sapiens_lite_host/torchscript/depth/checkpoints/sapiens_1b/sapiens_1b_render_people_epoch_88_torchscript.pt2",
        "2b": "/media/oem/12TB/sapiens/pretrain/sapiens_lite_host/torchscript/depth/checkpoints/sapiens_2b/sapiens_2b_render_people_epoch_25_torchscript.pt2",
    }
    checkpoint_path = CHECKPOINTS[model_size]
    model = torch.jit.load(checkpoint_path)
    model.eval()
    return model.to(device)

def estimate_depth_sapien(frames, model_size="1b", use_background_removal=False):
    """Use Sapiens pre-trained depth estimator."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_sapiens_model(model_size, device)
    
    if use_background_removal:
        seg_model = load_sapiens_model("fg-bg-1b", device)  # Load segmentation model
    
    transform = transforms.Compose([
        transforms.Resize((1024, 768)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], 
                             std=[58.5/255, 57.0/255, 57.5/255])
    ])
    
    depth_maps = []
    for frame in frames:
        print("frame:",frame)
        # Preprocess
        frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_pil = Image.fromarray(frame_np)
        frame_input = transform(frame_pil).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            depth = model(frame_input)
        
        # Resize back to original size
        depth = torch.nn.functional.interpolate(
            depth, size=frame.shape[1:], mode="bilinear", align_corners=False
        ).squeeze()
        
        if use_background_removal:
            with torch.no_grad():
                seg_output = seg_model(frame_input)
            seg_mask = (seg_output.argmax(dim=1) > 0).float()
            seg_mask = torch.nn.functional.interpolate(
                seg_mask.unsqueeze(1), size=frame.shape[1:], mode="nearest"
            ).squeeze()
            depth[seg_mask == 0] = float('nan')
        
        depth_maps.append(depth)
    
    return torch.stack(depth_maps)


# def estimate_depth(frames):
#     """Use a pre-trained monocular depth estimator (MiDaS)."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     midas = load_midas_model("DPT_Large", device)
    
#     depth_maps = []
#     for frame in frames:
#         # Preprocess
#         frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#         frame_pil = Image.fromarray(frame_np)
#         frame_input = Resize((384, 384))(frame_pil)
#         frame_input = torch.from_numpy(np.array(frame_input)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
#         # Inference
#         with torch.no_grad():
#             depth = midas(frame_input.to(device))
        
#         # Resize back to original size
#         depth = torch.nn.functional.interpolate(
#             depth.unsqueeze(1), size=frame.shape[1:], mode="bicubic", align_corners=False
#         ).squeeze()
        
#         depth_maps.append(depth)
    
#     return torch.stack(depth_maps)

def detect_and_track_humans_original(frames):
    """Use Detectron2 for human detection and a simple IoU-based tracking."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    
    human_masks = []
    prev_boxes = None
    
    for frame in frames:
        frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        outputs = predictor(frame_np)
        
        # Filter for human class (typically class 0 in COCO)
        human_indices = outputs["instances"].pred_classes == 0
        human_boxes = outputs["instances"].pred_boxes.tensor[human_indices]
        human_masks_frame = outputs["instances"].pred_masks[human_indices]
        
        if prev_boxes is not None:
            # Simple tracking based on IoU
            ious = torchvision.ops.box_iou(human_boxes, prev_boxes)
            matched_indices = ious.argmax(dim=1)
            human_masks_frame = human_masks_frame[matched_indices]
        
        prev_boxes = human_boxes
        human_masks.append(human_masks_frame.float())
    
    return torch.stack(human_masks)

def extract_pose(frames):
    """Use SMPL for pose estimation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl = SMPL("path/to/smpl/model", batch_size=1).to(device)
    
    # Note: Actual pose estimation from 2D images to SMPL parameters
    # is a complex task that typically requires a specialized model.
    # For simplicity, we'll return random SMPL parameters here.
    batch_size, _, height, width = frames.shape
    random_poses = torch.rand(batch_size, 72).to(device)  # 72 = 3 (global orient) + 69 (body pose)
    random_betas = torch.rand(batch_size, 10).to(device)  # 10 shape parameters
    
    smpl_output = smpl(body_pose=random_poses[:, 3:], global_orient=random_poses[:, :3], betas=random_betas)
    return smpl_output.vertices, smpl_output.joints

def inpaint_scene(scene_frames):
    """Use a simple image inpainting method to fill in missing areas."""
    inpainted_frames = []
    
    for frame in scene_frames:
        frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = np.all(frame_np == 0, axis=2).astype(np.uint8) * 255
        
        # Use OpenCV's inpainting
        inpainted = cv2.inpaint(frame_np, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        inpainted_frames.append(torch.from_numpy(inpainted).permute(2, 0, 1).float() / 255.0)
    
    return torch.stack(inpainted_frames)

def compute_masks(depth_maps: torch.Tensor, human_masks: torch.Tensor, sam_predictor: SAM2ImagePredictor,
                  depth_threshold: float = 0.5, 
                  smoothing_kernel_size: int = 5, 
                  min_area_ratio: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute masks for human, scene, and occlusion layers based on depth.

    Args:
        depth_maps (torch.Tensor): Depth maps of shape [B, H, W]
        human_masks (torch.Tensor): Human masks of shape [B, H, W]
        depth_threshold (float): Threshold for separating foreground and background
        smoothing_kernel_size (int): Size of the kernel for smoothing masks
        min_area_ratio (float): Minimum area ratio to keep a connected component

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Human, scene, and occlusion masks
    """
    device = depth_maps.device
    batch_size, height, width = depth_maps.shape

    # Normalize depth maps
    normalized_depth = (depth_maps - depth_maps.min(dim=(1,2), keepdim=True).values) / \
                       (depth_maps.max(dim=(1,2), keepdim=True).values - depth_maps.min(dim=(1,2), keepdim=True).values)

    # Compute initial scene and occlusion masks
    scene_mask = (~human_masks.bool()) & (normalized_depth > depth_threshold)
    occlusion_mask = (~human_masks.bool()) & (normalized_depth <= depth_threshold)

    # Smooth masks
    smoothing_kernel = torch.ones(1, 1, smoothing_kernel_size, smoothing_kernel_size, device=device) / (smoothing_kernel_size ** 2)
    scene_mask = F.conv2d(scene_mask.float().unsqueeze(1), smoothing_kernel, padding=smoothing_kernel_size//2).squeeze(1) > 0.5
    occlusion_mask = F.conv2d(occlusion_mask.float().unsqueeze(1), smoothing_kernel, padding=smoothing_kernel_size//2).squeeze(1) > 0.5

    # Remove small connected components
    scene_mask = remove_small_components(scene_mask, min_area_ratio)
    occlusion_mask = remove_small_components(occlusion_mask, min_area_ratio)

    # Ensure no overlap between masks
    overlap = scene_mask & occlusion_mask
    scene_mask = scene_mask & ~overlap
    occlusion_mask = occlusion_mask & ~overlap

    return human_masks, scene_mask.float(), occlusion_mask.float()

def remove_small_components(mask: torch.Tensor, min_area_ratio: float) -> torch.Tensor:
    """
    Remove small connected components from the mask.

    Args:
        mask (torch.Tensor): Binary mask of shape [B, H, W]
        min_area_ratio (float): Minimum area ratio to keep a connected component

    Returns:
        torch.Tensor: Cleaned mask
    """
    batch_size, height, width = mask.shape
    min_area = height * width * min_area_ratio

    cleaned_mask = torch.zeros_like(mask)
    for i in range(batch_size):
        labels, num_labels = label(mask[i].cpu().numpy())
        for j in range(1, num_labels + 1):
            if np.sum(labels == j) >= min_area:
                cleaned_mask[i][labels == j] = 1

    return cleaned_mask


# Example usage:
# video_path = "path/to/your/video.mp4"
# frames = load_video(video_path)
# depth_maps = estimate_depth(frames)
# human_masks = detect_and_track_humans(frames)
# vertices, joints = extract_pose(frames)
# scene_frames = frames * (~human_masks.bool()).float().unsqueeze(1)
# inpainted_scene = inpaint_scene(scene_frames)
# human_mask, scene_mask, occlusion_mask = compute_masks(depth_maps, human_masks)


# def process_video(video_path: str, config_path: str, checkpoint_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     frames = load_video(video_path)
#     depth_maps = estimate_depth_sapien(frames)
#     human_masks = detect_and_track_humans(frames)
    
#     human_mask, scene_mask, occlusion_mask = compute_masks(
#         depth_maps, 
#         human_masks, 
#         depth_threshold=0.5,  # Adjust as needed
#         smoothing_kernel_size=5,  # Adjust as needed
#         min_area_ratio=0.01  # Adjust as needed
#     )
    
#     # Areas to be inpainted are where the human or occlusion masks are 1
#     inpainting_mask = human_mask | occlusion_mask
    
#     scene_frames = frames * (~inpainting_mask)
#     inpainted_scene = inpaint_scene(scene_frames, inpainting_mask, config_path, checkpoint_path)
    
#     return inpainted_scene, human_mask * frames, occlusion_mask * frames

# # Usage
# video_path = "path/to/your/video.mp4"
# config_path = "path/to/lama/config.yaml"
# checkpoint_path = "path/to/lama/checkpoint.pth"
# inpainted_scene, human_video, occlusion_video = process_video(video_path, config_path, checkpoint_path)

# Update the process_video function
def process_video(video_path: str, config_path: str, checkpoint_path: str, sam_config: str, sam_checkpoint: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = load_video(video_path)
    depth_maps = estimate_depth_sapien(frames)
    
    # Initialize SAM 2 model
    sam_predictor = load_sam2_model(sam_config, sam_checkpoint)
    
    human_masks = detect_and_track_humans(frames, sam_predictor)
    
    human_mask, scene_mask, occlusion_mask = compute_masks(
        depth_maps, 
        human_masks,
        sam_predictor,
        depth_threshold=0.5,
        smoothing_kernel_size=5,
        min_area_ratio=0.01
    )
    
    # Areas to be inpainted are where the human or occlusion masks are 1
    inpainting_mask = human_mask | occlusion_mask
    
    scene_frames = frames * (~inpainting_mask)
    inpainted_scene = inpaint_scene(scene_frames, inpainting_mask, config_path, checkpoint_path)
    
    return inpainted_scene, human_mask * frames, occlusion_mask * frames


def load_sam2_model(config_file, checkpoint_path):
    """Load and initialize the SAM 2 model."""
    model = build_sam2(config_file, checkpoint_path)
    predictor = SAM2ImagePredictor(model)
    return predictor

def detect_and_track_humans(frames, sam_predictor):
    """Use SAM 2 for human detection and a simple IoU-based tracking."""
    human_masks = []
    prev_boxes = None
    
    for frame in frames:
        frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        sam_predictor.set_image(frame_np)
        
        # Use the center of the image as a prompt point
        h, w = frame_np.shape[:2]
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])  # 1 for foreground
        
        masks, scores, _ = sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        # Select the mask with the highest score
        best_mask = masks[np.argmax(scores)]
        
        # Convert mask to tensor
        mask_tensor = torch.from_numpy(best_mask).float()
        
        if prev_boxes is not None:
            # Simple tracking based on IoU
            current_box = mask_to_box(mask_tensor)
            iou = box_iou(current_box.unsqueeze(0), prev_boxes)
            if iou.max() > 0.5:  # You can adjust this threshold
                matched_index = iou.argmax()
                mask_tensor = human_masks[-1][matched_index]
        
        human_masks.append(mask_tensor.unsqueeze(0))
        prev_boxes = mask_to_box(mask_tensor).unsqueeze(0)
    
    return torch.cat(human_masks, dim=0)

def mask_to_box(mask):
    """Convert a binary mask to a bounding box."""
    y, x = torch.where(mask > 0.5)
    return torch.tensor([x.min(), y.min(), x.max(), y.max()])

def box_iou(box1, box2):
    """Compute IoU between two boxes."""
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



