import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class Config:
    CHECKPOINTS_DIR = '/media/oem/12TB/sapiens/pretrain/sapiens_lite_host/torchscript/depth/checkpoints'
    CHECKPOINTS = {
        "0.3b": "sapiens_0.3b_render_people_epoch_100_torchscript.pt2",
        "0.6b": "sapiens_0.6b_render_people_epoch_70_torchscript.pt2",
        "1b": "sapiens_1b_render_people_epoch_88_torchscript.pt2",
        "2b": "sapiens_2b_render_people_epoch_25_torchscript.pt2",
    }
    SEG_CHECKPOINTS = {
        "fg-bg-1b": "sapiens_1b_seg_foreground_epoch_8_torchscript.pt2",
        "part-seg-1b": "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
    }

class ModelManager:
    @staticmethod
    def load_model(checkpoint_name: str):
        checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, checkpoint_name)
        model = torch.jit.load(checkpoint_path)
        model.eval()
        model.to("cuda")
        return model

    @staticmethod
    @torch.inference_mode()
    def run_model(model, input_tensor, height, width):
        output = model(input_tensor)
        return torch.nn.functional.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], 
                                 std=[58.5/255, 57.0/255, 57.5/255])
        ])

    @torch.inference_mode()
    def estimate_depth(self, image: Image.Image, depth_model_name: str, seg_model_name: str):
        depth_model = ModelManager.load_model(Config.CHECKPOINTS[depth_model_name])
        seg_model = ModelManager.load_model(Config.SEG_CHECKPOINTS[seg_model_name]) if seg_model_name != "no-bg-removal" else None

        input_tensor = self.transform(image).unsqueeze(0).to("cuda")
        
        depth_output = ModelManager.run_model(depth_model, input_tensor, image.height, image.width)
        depth_map = depth_output.squeeze().cpu().numpy()

        if seg_model:
            seg_output = ModelManager.run_model(seg_model, input_tensor, image.height, image.width)
            seg_mask = (seg_output.argmax(dim=1) > 0).float().cpu().numpy()[0]
            depth_map[seg_mask == 0] = np.nan

        return depth_map

    @staticmethod
    def colorize_depth_map(depth_map):
        depth_foreground = depth_map[~np.isnan(depth_map)]
        if len(depth_foreground) > 0:
            min_val, max_val = np.nanmin(depth_foreground), np.nanmax(depth_foreground)
            depth_normalized = (depth_map - min_val) / (max_val - min_val)
            depth_normalized = 1 - depth_normalized
            depth_normalized = np.nan_to_num(depth_normalized, nan=0)
            cmap = plt.get_cmap('inferno')
            depth_colored = (cmap(depth_normalized) * 255).astype(np.uint8)[:, :, :3]
        else:
            depth_colored = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
        return depth_colored

def main():
    parser = argparse.ArgumentParser(description="Depth Estimation Inference")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("output_image", help="Path to save the output image")
    parser.add_argument("output_npy", help="Path to save the output depth map as .npy file")
    parser.add_argument("--depth_model", choices=list(Config.CHECKPOINTS.keys()), default="1b", help="Depth model size")
    parser.add_argument("--seg_model", choices=list(Config.SEG_CHECKPOINTS.keys()) + ["no-bg-removal"], default="fg-bg-1b", help="Segmentation model for background removal")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    image_processor = ImageProcessor()

    input_image = Image.open(args.input).convert("RGB")
    depth_map = image_processor.estimate_depth(input_image, args.depth_model, args.seg_model)
    
    # Save depth map as .npy file
    np.save(args.output_npy, depth_map)
    
    # Colorize and save depth map as image
    depth_colored = image_processor.colorize_depth_map(depth_map)
    Image.fromarray(depth_colored).save(args.output_image)

    print(f"Results saved to {args.output_image} and {args.output_npy}")

if __name__ == "__main__":
    main()