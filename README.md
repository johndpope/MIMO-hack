# MIMO-hack


https://github.com/Uminosachi/inpaint-anything

```shell
test_motion.py
```
https://huggingface.co/lilpotat/pytorch3d/tree/main


## Dataset
We create a human video dataset called HUD-7K
to train our model. This dataset consists of 5K real character videos and 2K synthetic character animations. The
former does not require any annotations and can be automatically decomposed to various spatial attributes via our
scheme. To enlarge the range of the real dataset, we also
synthesize 2K videos by rendering character animations in
complex motions under multiple camera views, utilizing En3D [21]. 
These synthetic videos are equipped with accurate annotations due to completely controlled production.

https://github.com/menyifang/En3D



## Synthetic Training data
https://openxlab.org.cn/datasets/OpenXDLab/SynBody

```
pip install openxlab #Install

pip install -U openxlab #Upgrade

openxlab login # Log in and enter the corresponding AK/SK. Please view AK/SK at usercenter

openxlab dataset info --dataset-repo OpenXDLab/SynBody # Dataset information viewing and View Dataset File List

openxlab dataset get --dataset-repo OpenXDLab/SynBody #Dataset download

openxlab dataset download --dataset-repo OpenXDLab/SynBody --source-path /README.md --target-path /path/to/local/folder #Dataset file download
```



## Sapiens - get image - produce depths / normals / pose
https://github.com/facebookresearch/sapiens
```shell
python pose_vis.py '/home/oem/Desktop/image_1.png'  test.png output.json
python normal_vis.py '/home/oem/Desktop/image_1.png'  test.png 
python depth_estimation.py input_image.png output_depth_image.png output_depth_map.npy --depth_model 1b --seg_model fg-bg-1b

```


## LAMA - SOTA inpainting 
https://github.com/advimman/lama



# Todo Components  

## 1. Setup and Dependencies
- [x] Import necessary libraries for deep learning and 3D processing
  - `torch`, `torchvision`, `pytorch3d`, `clip`, `SMPL`, `diffusers`
  - Custom utility functions for video loading, depth estimation, and mask computation

## 2. Define Model Components

### 2.1 Temporal Attention Layer
- [x] Implement temporal attention layer for improved 3D motion representation
  - Utilize `GroupNorm` and `BasicTransformerBlock` from diffusers

### 2.2 Differentiable Rasterizer
- [x] Define `DifferentiableRasterizer` for projecting 3D models into 2D feature maps
  - Use `pytorch3d` for rendering and rasterization
  - Integrate `PerspectiveCameras` for basic 3D-to-2D projection

### 2.3 Structured Motion Encoder
- [x] Create `StructuredMotionEncoder` for encoding SMPL-based human motion
  - Load SMPL model for 3D human body modeling
  - Project SMPL vertices onto 2D plane
  - Encode motion using 3D CNN layers

### 2.4 Canonical Identity Encoder
- [x] Develop `CanonicalIdentityEncoder` for disentangling identity attributes
  - Use `CLIP` model for global and local feature extraction
  - Add a custom reference network for additional local features

### 2.5 Scene and Occlusion Encoder
- [x] Implement `SceneOcclusionEncoder` using a shared pre-trained VAE for encoding
  - Add temporal convolution layers for time-series input processing

### 2.6 Diffusion Decoder
- [x] Design the `DiffusionDecoder` using `UNet3DConditionModel` from diffusers
  - Customize with temporal attention blocks and stable diffusion-based decoder

### 2.7 MIMO Model
- [x] Combine all components into a unified `MIMOModel`
  - Motion encoder, identity encoder, and scene/occlusion encoder as input conditions
  - Use diffusion decoder for video synthesis from latent representations

## 3. Dataset Handling

### 3.1 Dataset Class
- [x] Create `MIMODataset` class for loading video data and corresponding attributes
  - Load video frames, SMPL parameters, identity images, and camera parameters
  - Use LAMA inpainting for scene reconstruction

### 3.2 Data Preprocessing
- [x] Implement functions for data preprocessing
  - Load video frames as tensors
  - Estimate depth using pre-trained MiDaS or Sapiens depth estimator
  - Detect and track humans using `Detectron2`
  - Extract SMPL parameters for human pose estimation
  - Inpaint scene using OpenCV inpainting

### 3.3 Mask Computation
- [x] Compute masks for human, scene, and occlusion layers
  - Use depth maps and human masks for spatial decomposition
  - Remove small components in masks using connected component analysis

## 4. Training Procedure

### 4.1 Forward Diffusion Sampling
- [x] Define the forward diffusion process with noise scheduling
  - Implement the `linear_beta_schedule` for noise addition

### 4.2 Training Loop
- [x] Implement the main training loop for the MIMO model
  - Load data from `MIMODataset`
  - Apply noise to latent codes and predict the noise residual using the diffusion decoder
  - Optimize with MSE loss

## 5. Inference Pipeline
- [x] Implement the inference function for generating new character videos
  - Start from random noise and use DDIM scheduler for step-wise denoising
  - Encode conditions (identity, motion, scene) and generate a video from latent space

## 6. Hyperparameters and Configuration
- [x] Define hyperparameters for training and inference
  - Number of vertices, feature dimension, image size, timesteps, batch size, learning rate

## 7. Additional Components
- [x] Add utility functions for processing video frames, depth maps, and masks
  - `load_video`, `estimate_depth`, `detect_and_track_humans`, `inpaint_scene`, `compute_masks`

## 8. Saving and Loading the Model
- [x] Save trained model weights using `torch.save`
- [x] Load the saved model for inference

