import logging
import numpy as np
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

from detectron2.data import transforms as T
from .transforms.custom_augmentation_impl import EfficientDetResizeCrop

def build_custom_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    if cfg.INPUT.CUSTOM_AUG == 'ResizeShortestEdge':
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    elif cfg.INPUT.CUSTOM_AUG == 'EfficientDetResizeCrop':
        if is_train:
            scale = cfg.INPUT.SCALE_RANGE
            size = cfg.INPUT.TRAIN_SIZE
        else:
            scale = (1, 1)
            size = cfg.INPUT.TEST_SIZE
        augmentation = [EfficientDetResizeCrop(size, scale)]
    else:
        assert 0, cfg.INPUT.CUSTOM_AUG

    if is_train:
        #augmentation.append(T.RandomFlip())
        augmentation.append(T.RandomContrast(0.2, 1.8))
        augmentation.append(T.RandomBrightness(0.5, 1.5))
        augmentation.append(T.RandomSaturation(0.9, 1.1))
        augmentation.append(T.RandomLighting(0.5))

        augmentation.append(T.RandomRotation(cfg.INPUT.ROTATION_ANGLES, expand=False, sample_style="choice"))
        augmentation.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
        augmentation.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))
    return augmentation


build_custom_transform_gen = build_custom_augmentation
"""
Alias for backward-compatibility.
"""
