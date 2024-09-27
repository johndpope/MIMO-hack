from typing import List, Optional, Sequence, Union

import torch
import cv2
import numpy as np
from mmcv.ops import RoIPool
from mmengine.dataset import Compose, pseudo_collate
from mmengine.device import get_device
from mmengine.registry import init_default_scope
from mmdet.apis import inference_detector, init_detector
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import get_test_pipeline_cfg


ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]

def nms(dets: np.ndarray, thr: float):
    """Greedily select boxes with high confidence and overlap <= thr.
    Args:
        dets (np.ndarray): [[x1, y1, x2, y2, score]].
        thr (float): Retain overlap < thr.
    Returns:
        list: Indexes to keep.
    """
    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]

    return keep

def adapt_mmdet_pipeline(cfg):
    """Converts pipeline types in MMDetection's test dataloader to use the
    'mmdet' namespace.

    Args:
        cfg (ConfigDict): Configuration dictionary for MMDetection.

    Returns:
        ConfigDict: Configuration dictionary with updated pipeline types.
    """
    # use lazy import to avoid hard dependence on mmdet
    from mmdet.datasets import transforms

    if 'test_dataloader' not in cfg:
        return cfg

    pipeline = cfg.test_dataloader.dataset.pipeline
    for trans in pipeline:
        if trans['type'] in dir(transforms):
            trans['type'] = 'mmdet.' + trans['type']

    return cfg


def inference_detector(
    model: torch.nn.Module,
    imgs: ImagesType,
    test_pipeline: Optional[Compose] = None,
    text_prompt: Optional[str] = None,
    custom_entities: bool = False,
) -> Union[DetDataSample, SampleList]:
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    if isinstance(imgs, torch.Tensor):
        if imgs.is_cuda:
            imgs = imgs.cpu()

        # Remove batch dimension and transpose
        imgs = imgs.squeeze(0).permute(1, 2, 0).numpy()
        
        # Ensure the data type is appropriate (uint8 for most image processing functions)
        imgs = (imgs * 255).astype(np.uint8)

    if isinstance(imgs, (list, tuple)) or (isinstance(imgs, np.ndarray) and len(imgs.shape) == 4):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = "mmdet.LoadImageFromNDArray"

        test_pipeline = Compose(test_pipeline)

    if model.data_preprocessor.device.type == "cpu":
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), "CPU inference with RoIPool is not supported currently."

    result_list = []
    for i, img in enumerate(imgs):
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)

        if text_prompt:
            data_["text"] = text_prompt
            data_["custom_entities"] = custom_entities

        # build the data pipeline
        data_ = test_pipeline(data_)

        data_["inputs"] = [data_["inputs"]]
        data_["data_samples"] = [data_["data_samples"]]

        # forward the model
        with torch.no_grad(), torch.autocast(device_type=get_device(), dtype=torch.bfloat16):
            results = model.test_step(data_)[0]

        result_list.append(results)

    if not is_batch:
        return result_list[0]
    else:
        return result_list


def process_one_image_bbox(pred_instance, det_cat_id, bbox_thr, nms_thr):
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
    )
    bboxes = bboxes[
        np.logical_and(
            pred_instance.labels == det_cat_id,
            pred_instance.scores > bbox_thr,
        )
    ]
    bboxes = bboxes[nms(bboxes, nms_thr), :4]
    return bboxes


def process_images_detector(imgs, detector):
    """Visualize predicted keypoints (and heatmaps) of one image."""
    # predict bbox
    det_results = inference_detector(detector, imgs)
    pred_instances = list(
        map(lambda det_result: det_result.pred_instances.numpy(), det_results)
    )
    bboxes_batch = list(
        map(
            lambda pred_instance: process_one_image_bbox(
                pred_instance, 0, 0.3, 0.3 ## argparse.Namespace(det_cat_id=0, bbox_thr=0.3, nms_thr=0.3),
            ),
            pred_instances,
        )
    )

    return bboxes_batch