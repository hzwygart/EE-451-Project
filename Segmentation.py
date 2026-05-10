import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2hsv
import cv2
from typing import Callable
from skimage.morphology import closing, opening, disk, remove_small_holes, remove_small_objects

def extract_hsv_channels(img):
    """
    Extract HSV channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    data_h: np.ndarray (M, N)
        Hue channel of input image
    data_s: np.ndarray (M, N)
        Saturation channel of input image
    data_v: np.ndarray (M, N)
        Value channel of input image
    """
    M, N, C = np.shape(img)

    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    img_hsv = rgb2hsv(img)
    data_h = img_hsv[:, :, 0]
    data_s = img_hsv[:, :, 1]
    data_v = img_hsv[:, :, 2]

    return data_h, data_s, data_v

def apply_hsv_threshold(img):
    """
    Apply color-specific HSV thresholds to detect card and token regions.
    Thresholds derived from HSV scatter plot analysis, tuned on both
    white background (L1000849) and noisy background (L1000902).

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    img_th: np.ndarray (M, N)
        Raw binary mask before morphology.
    """
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    data_h, data_s, data_v = extract_hsv_channels(img=img)

    # Red: wraps around 0 and 1, strict V to exclude pinkish leaves
    red_mask = (
        ((data_h < 0.05) | (data_h > 0.95)) &
        (data_s > 0.65) &
        (data_v > 0.85)  # tightened from 0.80
    )

    # Yellow: narrow hue band, high V
    yellow_mask = (
        (data_h > 0.10) & (data_h < 0.16) &
        (data_v > 0.60)
    )

    # Green: strict V range to exclude dark leaf greens
    green_mask = (
        (data_h > 0.25) & (data_h < 0.42) &
        (data_v > 0.60) & (data_v < 0.88) &  # tightened from 0.55
        (data_s > 0.20)
    )

    # Blue: tight hue, high saturation
    blue_mask = (
        (data_h > 0.50) & (data_h < 0.65) &
        (data_s > 0.40) &
        (data_v > 0.30)
    )

    # Token + wild/draw_4: dark regions
    dark_mask = (
        (data_v < 0.55) &
        (data_s < 0.31) &
        (data_h < 0.12)
    )

    # Combine
    img_th = red_mask | yellow_mask | green_mask | blue_mask | dark_mask

    # Global saturation filter — kills leaves, protects dark mask
    high_sat = data_s > 0.46
    img_th = img_th & (high_sat | dark_mask)

    return img_th


def apply_morphology(img_th):
    """
    Apply morphological operations to clean up the binary mask.
    Removes noise, fills card regions, and removes small objects.

    Args
    ----
    img_th: np.ndarray (M, N)
        Raw binary mask from apply_hsv_threshold.

    Return
    ------
    img_th: np.ndarray (M, N)
        Cleaned binary mask after morphology.
    """
    img_th = remove_small_objects(img_th.astype(bool), min_size=13)
    img_th = closing(img_th.astype(bool), disk(9))
    img_th = opening(img_th.astype(bool), disk(6))
    img_th = remove_small_objects(img_th.astype(bool), min_size=6000)
    img_th = remove_small_holes(img_th.astype(bool), area_threshold=5000)

    return img_th

def segment_image(img):
    """
    Full segmentation pipeline: HSV threshold + morphology.
    Args
    ----
    img: np.ndarray (M, N, C)
        Input RGB image.

    Return
    ------
    mask: np.ndarray (M, N)
        Clean binary mask of card and token regions.
    """
    img_th = apply_hsv_threshold(img)
    mask = apply_morphology(img_th)
    return mask