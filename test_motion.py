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
    npz = np.load(smpl_param_ckpt_fpath)
    d = {x: torch.from_numpy(npz[x]).float() for x in npz.keys()}

    betas = d['betas']
    orient = d['global_orient']
    body_pose = d['body_pose']
    jaw_pose = d['jaw_pose']
    leye_pose = torch.zeros_like(jaw_pose)
    reye_pose = torch.zeros_like(jaw_pose)
    left_hand_pose = d['left_hand_pose']
    right_hand_pose = d['right_hand_pose']
    poses = torch.cat([orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose], dim=1)
    if body_only:
        poses[:, body_pose.shape[1]:] *= 0.0
    trans = d['transl']
    return betas, poses, trans

def test_motion_encoder(data_dir, cam_ids_to_use):
    # Initialize the StructuredMotionEncoder
    num_vertices = 6890  # SMPL model typically has 6890 vertices
    feature_dim = 64  # Adjust as needed
    image_size = 256  # Adjust based on your rendered image size
    motion_encoder = StructuredMotionEncoder(num_vertices, feature_dim, image_size)

    # Load camera parameters
    cam_data = get_cams(data_dir)
    print("cam_data:",cam_data)
    cam_ssn_list = get_cam_ssns(data_dir)

    # Load SMPLX parameters
    betas, poses, trans = load_smplx_params(os.path.join(data_dir, 'smpl_params.npz'))

    # Prepare camera parameters
    camera_params = []
    for cam_id in cam_ids_to_use:
        cam_ssn = cam_ssn_list[cam_id]
        R = torch.tensor(cam_data[cam_ssn]['R'], dtype=torch.float32).flatten()
        T = torch.tensor(cam_data[cam_ssn]['T'], dtype=torch.float32)
        camera_params.append(torch.cat([R, T]))
    camera_params = torch.stack(camera_params)

    # Ensure the data is in the correct shape
    num_frames = poses.shape[0]
    num_cameras = len(cam_ids_to_use)
    print("num_cameras:", num_cameras)

    # Repeat camera params for each frame
    camera_params = camera_params.unsqueeze(0).repeat(num_frames, 1, 1)

    # Combine poses and trans
    smplx_params = torch.cat([poses, trans], dim=-1)

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
    data_dir = '/media/oem/12TB/meshavatar/avatarrex_zzr'
    cam_ids_to_use = [0]  # Adjust this list as needed
    test_motion_encoder(data_dir, cam_ids_to_use)