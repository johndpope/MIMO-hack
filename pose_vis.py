import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import json

from classes_and_palettes import (
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
    GOLIATH_KEYPOINTS
)

from detector_utils import (
    adapt_mmdet_pipeline,
    init_detector,
    process_images_detector,
)

class Config:
    CHECKPOINTS_DIR = '/media/oem/12TB/sapiens/pretrain/sapiens_lite_host/torchscript/pose/checkpoints'
    CHECKPOINTS = {
        "1b": "sapiens_1b/sapiens_1b_goliath_best_goliath_AP_640_torchscript.pt2",
    }
    DETECTION_CHECKPOINT = '/media/oem/12TB/sapiens/pretrain/sapiens_host/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
    DETECTION_CONFIG = '/media/oem/12TB/sapiens/pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py'


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
    def run_model(model, input_tensor):
        return model(input_tensor)

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], 
                                 std=[58.5/255, 57.0/255, 57.5/255])
        ])
        self.detector = init_detector(
            Config.DETECTION_CONFIG, Config.DETECTION_CHECKPOINT, device='cuda'
        )
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

    def detect_persons(self, image: Image.Image):
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        bboxes_batch = process_images_detector(image, self.detector)
        bboxes = self.get_person_bboxes(bboxes_batch[0])
        return bboxes
    
    def get_person_bboxes(self, bboxes_batch, score_thr=0.3):
        person_bboxes = []
        for bbox in bboxes_batch:
            if len(bbox) == 5 and bbox[4] > score_thr:
                person_bboxes.append(bbox)
            elif len(bbox) == 4:
                person_bboxes.append(bbox + [1.0])
        return person_bboxes

    @torch.inference_mode()
    def estimate_pose(self, image: Image.Image, bboxes, model_name: str, kpt_threshold: float):
        pose_model = ModelManager.load_model(Config.CHECKPOINTS[model_name])
        
        result_image = image.copy()
        all_keypoints = []

        for bbox in bboxes:
            cropped_img = self.crop_image(result_image, bbox)
            input_tensor = self.transform(cropped_img).unsqueeze(0).to("cuda")
            heatmaps = ModelManager.run_model(pose_model, input_tensor)
            keypoints = self.heatmaps_to_keypoints(heatmaps[0].cpu().numpy())
            all_keypoints.append(keypoints)
            result_image = self.draw_keypoints(result_image, keypoints, bbox, kpt_threshold)
        
        return result_image, all_keypoints

    def process_image(self, image: Image.Image, model_name: str, kpt_threshold: float):
        bboxes = self.detect_persons(image)
        result_image, keypoints = self.estimate_pose(image, bboxes, model_name, kpt_threshold)
        return result_image, keypoints

    def crop_image(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox[:4])
        return image.crop((x1, y1, x2, y2))

    @staticmethod
    def heatmaps_to_keypoints(heatmaps):
        num_joints = heatmaps.shape[0]
        keypoints = {}
        for i, name in enumerate(GOLIATH_KEYPOINTS):
            if i < num_joints:
                heatmap = heatmaps[i]
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                conf = heatmap[y, x]
                keypoints[name] = (float(x), float(y), float(conf))
        return keypoints

    @staticmethod
    def draw_keypoints(image, keypoints, bbox, kpt_threshold):
        image = np.array(image)
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_size = np.sqrt(bbox_width * bbox_height)
        
        radius = max(1, int(bbox_size * 0.006))
        thickness = max(1, int(bbox_size * 0.006))
        bbox_thickness = max(1, thickness//4)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), bbox_thickness)
        
        for i, (name, (x, y, conf)) in enumerate(keypoints.items()):
            if conf > kpt_threshold and i < len(GOLIATH_KPTS_COLORS):
                x_coord = int(x * bbox_width / 192) + x1
                y_coord = int(y * bbox_height / 256) + y1
                color = GOLIATH_KPTS_COLORS[i]
                cv2.circle(image, (x_coord, y_coord), radius, color, -1)

        for _, link_info in GOLIATH_SKELETON_INFO.items():
            pt1_name, pt2_name = link_info['link']
            color = link_info['color']
            
            if pt1_name in keypoints and pt2_name in keypoints:
                pt1 = keypoints[pt1_name]
                pt2 = keypoints[pt2_name]
                if pt1[2] > kpt_threshold and pt2[2] > kpt_threshold:
                    x1_coord = int(pt1[0] * bbox_width / 192) + x1
                    y1_coord = int(pt1[1] * bbox_height / 256) + y1
                    x2_coord = int(pt2[0] * bbox_width / 192) + x1
                    y2_coord = int(pt2[1] * bbox_height / 256) + y1
                    cv2.line(image, (x1_coord, y1_coord), (x2_coord, y2_coord), color, thickness=thickness)

        return Image.fromarray(image)

def main():
    parser = argparse.ArgumentParser(description="Pose Estimation Inference")
    parser.add_argument("input", help="Path to the input image or JSON file")
    parser.add_argument("output_image", help="Path to save the output image")
    parser.add_argument("--output_json", help="Path to save the output JSON (only used with image input)")
    parser.add_argument("--model", choices=list(Config.CHECKPOINTS.keys()), default="1b", help="Model size")
    parser.add_argument("--kpt_threshold", type=float, default=0.3, help="Keypoint confidence threshold")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    image_processor = ImageProcessor()

    input_image = Image.open(args.input).convert("RGB")
    result_image, keypoints = image_processor.process_image(input_image, args.model, args.kpt_threshold)
    result_image.save(args.output_image)
    
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(keypoints, f, indent=2)
        print(f"Results saved to {args.output_image} and {args.output_json}")
    else:
        print(f"Results saved to {args.output_image}")

if __name__ == "__main__":
    main()
    # python pose_vis.py '/home/oem/Desktop/image_1.png'  test.png output.json
