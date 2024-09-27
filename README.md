# MIMO-hack





## Dataset
We create a human video dataset called HUD-7K
to train our model. This dataset consists of 5K real character videos and 2K synthetic character animations. The
former does not require any annotations and can be automatically decomposed to various spatial attributes via our
scheme. To enlarge the range of the real dataset, we also
synthesize 2K videos by rendering character animations in
complex motions under multiple camera views, utilizing En3D [21]. 
These synthetic videos are equipped with accurate annotations due to completely controlled production.

https://github.com/menyifang/En3D


```shell
python pose_vis.py '/home/oem/Desktop/image_1.png'  test.png output.json
python normal_vis.py '/home/oem/Desktop/image_1.png'  test.png 
python depth_estimation.py input_image.png output_depth_image.png output_depth_map.npy --depth_model 1b --seg_model fg-bg-1b

```
