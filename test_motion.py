import torch
import json
import numpy as np
import os
from Model import StructuredMotionEncoder

def get_cams(root_dir):
    with open(os.path.join(root_dir, 'calibration_full.json'), 'r') as fp:
        cam_data = json.load(fp)
    return cam_data

def get_cam_ssns(dataset_fpath):
    cam_ssns = []
    with open(os.path.join(dataset_fpath, 'cam_ssns.txt'), 'r') as fp:
        lns = fp.readlines()
    for ln in lns:
        ln = ln.strip().split(' ')
        if len(ln) > 0:
            cam_ssns.append(ln[0])
    return cam_ssns

def load_smplx_params(smpl_param_ckpt_fpath, body_only=True):
    print(f"Loading SMPLX parameters from: {smpl_param_ckpt_fpath}")
    npz = np.load(smpl_param_ckpt_fpath)
    print(f"Loaded file keys: {npz.keys()}")

    d = {x: torch.from_numpy(npz[x]).float() for x in npz.keys()}
    
    print(f"Converted numpy arrays to torch tensors.")

    betas = d['betas']
    orient = d['global_orient']
    body_pose = d['body_pose']
    
    print(f"Initial shapes:")
    print(f"  betas: {betas.shape}")
    print(f"  orient: {orient.shape}")
    print(f"  body_pose: {body_pose.shape}")

    jaw_pose = d['jaw_pose'] if 'jaw_pose' in d else torch.zeros_like(orient)
    leye_pose = d['leye_pose'] if 'leye_pose' in d else torch.zeros_like(orient)
    reye_pose = d['reye_pose'] if 'reye_pose' in d else torch.zeros_like(orient)
    left_hand_pose = d['left_hand_pose'] if 'left_hand_pose' in d else torch.zeros_like(orient).repeat(1, 15)
    right_hand_pose = d['right_hand_pose'] if 'right_hand_pose' in d else torch.zeros_like(orient).repeat(1, 15)
    
    print(f"Jaw pose shape: {jaw_pose.shape}")
    print(f"Left eye pose shape: {leye_pose.shape}")
    print(f"Right eye pose shape: {reye_pose.shape}")
    print(f"Left hand pose shape: {left_hand_pose.shape}")
    print(f"Right hand pose shape: {right_hand_pose.shape}")

    # Get the number of frames from the orient shape
    num_frames = orient.shape[0]

    # Ensure all pose parameters have the correct shape
    orient = orient.view(num_frames, 3)
    body_pose = body_pose.view(num_frames, -1)  # Keep the original number of parameters
    jaw_pose = jaw_pose.view(num_frames, 3)
    leye_pose = leye_pose.view(num_frames, 3)
    reye_pose = reye_pose.view(num_frames, 3)
    left_hand_pose = left_hand_pose.view(num_frames, 45)  # 15 joints * 3
    right_hand_pose = right_hand_pose.view(num_frames, 45)  # 15 joints * 3

    print(f"Reshaped parameters:")
    print(f"  orient: {orient.shape}")
    print(f"  body_pose: {body_pose.shape}")
    print(f"  jaw_pose: {jaw_pose.shape}")
    print(f"  leye_pose: {leye_pose.shape}")
    print(f"  reye_pose: {reye_pose.shape}")
    print(f"  left_hand_pose: {left_hand_pose.shape}")
    print(f"  right_hand_pose: {right_hand_pose.shape}")

    # Combine all pose parameters
    poses = torch.cat([orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose], dim=1)
    print(f"Combined pose shape: {poses.shape}")

    # if body_only:
    #     print(f"Zeroing out non-body poses beyond index 72.")
    #     poses[:, 72:] *= 0.0  # Zero out non-body poses
    
    trans = d['transl']
    print(f"Translational parameter shape: {trans.shape}")
    
    print(f"Final SMPLX parameters shapes:")
    print(f"  betas: {betas.shape}")
    print(f"  poses (combined): {poses.shape}")
    print(f"  trans: {trans.shape}")
    
    return betas, poses, trans


def test_motion_encoder(data_dir, cam_ids_to_use):
    print("Initializing StructuredMotionEncoder...")
    num_vertices = 10475  # SMPL-X has 10475 vertices
    feature_dim = 32
    image_size = 128
    motion_encoder = StructuredMotionEncoder(num_vertices, feature_dim, image_size)
    motion_encoder = motion_encoder.to('cuda')

    print("Loading camera parameters...")
    cam_data = get_cams(data_dir)
    print("Camera data:", cam_data)
    cam_ssn_list = get_cam_ssns(data_dir)

    print("Loading SMPLX parameters...")
    betas, poses, trans = load_smplx_params(os.path.join(data_dir, 'smpl_params.npz'))
    print(f"Loaded SMPLX parameters shapes: betas={betas.shape}, poses={poses.shape}, trans={trans.shape}")

    print("Preparing camera parameters...")
    camera_params = []
    for cam_id in cam_ids_to_use:
        cam_ssn = cam_ssn_list[cam_id]
        cam_info = cam_data[cam_ssn]
        R = torch.tensor(cam_info['R'], dtype=torch.float32).flatten()
        T = torch.tensor(cam_info['T'], dtype=torch.float32)
        
        # Extract intrinsics from 'K' matrix
        K = cam_info['K']
        fx = torch.tensor([K[0]], dtype=torch.float32)
        fy = torch.tensor([K[4]], dtype=torch.float32)
        cx = torch.tensor([K[2]], dtype=torch.float32)
        cy = torch.tensor([K[5]], dtype=torch.float32)
    
        # Concatenate all parameters for this camera
        cam_params = torch.cat([R, T, fx, fy, cx, cy])
        camera_params.append(cam_params)

    camera_params = torch.stack(camera_params)
    print(f"Camera parameters shape: {camera_params.shape}")

    num_frames = 10# just do 2 frames per camera poses.shape[0]
    num_cameras = len(cam_ids_to_use)
    print(f"Number of frames: {num_frames}")
    print(f"Number of cameras: {num_cameras}")

    print("Combining poses and trans...")
    smplx_params = torch.cat([poses, trans], dim=-1)
    print(f"SMPLX parameters shape: {smplx_params.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Moving tensors to device: {device}")
    smplx_params = smplx_params.to(device)
    camera_params = camera_params.to(device)
    betas = betas.to(device)

    # Process frames in batches for each camera
    batch_size = 8
    motion_codes_per_camera = []

    for cam_idx in range(num_cameras):
        print(f"Processing camera {cam_idx + 1}/{num_cameras}")
        cam_motion_codes = []

        for start_idx in range(0, num_frames, batch_size):
            end_idx = min(start_idx + batch_size, num_frames)
            batch_smplx_params = smplx_params[start_idx:end_idx].unsqueeze(0)
            batch_camera_params = camera_params[cam_idx].unsqueeze(0).unsqueeze(0).repeat(1, end_idx - start_idx, 1)

            # Forward pass
            frame_motion_code = motion_encoder(betas, batch_smplx_params, batch_camera_params)
            cam_motion_codes.append(frame_motion_code)

        # Combine motion codes for this camera
        cam_motion_codes = torch.cat(cam_motion_codes, dim=2)  # Shape: [1, code_dim, num_frames]
        motion_codes_per_camera.append(cam_motion_codes)

    # Combine motion codes for all cameras
    all_motion_codes = torch.cat(motion_codes_per_camera, dim=0)  # Shape: [num_cameras, code_dim, num_frames]

    print(f"Final motion codes shape: {all_motion_codes.shape}")
    return all_motion_codes
        

if __name__ == "__main__":
    data_dir = '/media/oem/12TB/meshavatar/avatarrex_zzr'
    cam_ids_to_use = [0,1,2,3,4,5,6,8,9,10,11,12,14,15]
    test_motion_encoder(data_dir, cam_ids_to_use)