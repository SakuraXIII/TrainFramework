import json
import logging
import math
import os.path
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
import paddle
from colorama import init, Fore, Style
from matplotlib import image as mpimg, pyplot as plt

import paddle.nn.functional as F

if TYPE_CHECKING:  # TYPE_CHECKING 在运行时为 False，因此不会导入包，避免内存占用。通常用于开发过程中引用包中的类型定义，且运行时又不需要该包
    from paddle import Tensor  # 用于开发时提供类型注解


def resize_and_pad(image: Tensor, target_size: Tuple[int, int]):
    """将图像缩放并填充到指定大小，同时记录缩放比例和填充信息。

    Args:
        image: 输入图像
        target_size: 目标大小 (height, width)
    Returns:
        Tuple[List, List, List]: 处理后的图像, 缩放比例, 填充信息
    """
    _, H, W = image.shape
    target_h, target_w = target_size
    # 计算缩放比例
    ratio = min(target_w / W, target_h / H)
    new_h, new_w = int(H * ratio), int(W * ratio)
    # 缩放图像
    resized_image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode='bilinear',
                                  align_corners=False).squeeze(0)
    # 计算填充
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    # 填充图像
    padded_image = F.pad(resized_image, (pad_left, pad_right, pad_top, pad_bottom), value=0)
    pad_info = (pad_top, pad_bottom, pad_left, pad_right)
    return padded_image, ratio, pad_info


def inverse_resize_and_pad(image: Tensor, ratio: float, pad_info: Tuple[int, int, int, int]):
    """根据缩放比例和填充信息，逆向恢复图像到原始大小。

    Args:
        image: 输入图像
        ratio: 缩放比例
        pad_info: 填充信息 (top, bottom, left, right)
    Returns:
        Tensor: 恢复后的图像
    """
    pad_top, pad_bottom, pad_left, pad_right = pad_info
    _, padded_h, padded_w = image.shape
    # 移除填充
    unpadded_image = image[:, pad_top:padded_h - pad_bottom, pad_left:padded_w - pad_right]
    # 恢复到原始大小
    original_h, original_w = int(unpadded_image.shape[1] / ratio), int(unpadded_image.shape[2] / ratio)
    restored_image = F.interpolate(unpadded_image.unsqueeze(0), size=(original_h, original_w), mode='bilinear',
                                   align_corners=False).squeeze(0)
    return restored_image


def image_to_patches(image, patch_size):
    """将图像划分成指定大小的patch，并对不足大小的patch用0填充。

    :param image: 输入图像 (C, H, W)
    :param patch_size: patch的大小 (height, width)
    :return: 划分后的patches (N, C, patch_size[0], patch_size[1]), 原图尺寸 (H, W)
    """
    _, H, W = image.shape
    patch_H, patch_W = patch_size
    
    # 计算补齐的大小
    pad_H = (patch_H - H % patch_H) % patch_H
    pad_W = (patch_W - W % patch_W) % patch_W
    
    # 补齐图像
    padded_image = F.pad(image, (0, pad_W, 0, pad_H), value=0)
    
    # 重新获取补齐后的图像大小
    _, padded_H, padded_W = padded_image.shape
    
    # 将图像划分成patch
    padded_image = padded_image.unfold(1, patch_H, patch_H)
    patches = padded_image.unfold(2, patch_W, patch_W)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    patches = patches.view(-1, image.size(0), patch_H, patch_W)
    
    return patches, (H, W)


def patches_to_image(patches, image_size, patch_size):
    """将patches拼接回原图。

    :param patches: 输入patches (N, C, patch_size[0], patch_size[1])
    :param image_size: 原图尺寸 (H, W)
    :param patch_size: patch的大小 (height, width)
    :return: 拼接后的图像 (C, H, W)
    """
    H, W = image_size
    patch_H, patch_W = patch_size
    
    # 计算补齐后的大小
    pad_H = (patch_H - H % patch_H) % patch_H
    pad_W = (patch_W - W % patch_W) % patch_W
    
    # 补齐后的图像大小
    padded_H = H + pad_H
    padded_W = W + pad_W
    
    # 恢复图像
    patches = patches.view(padded_H // patch_H, padded_W // patch_W, patches.size(1), patch_H, patch_W)
    patches = patches.permute(2, 0, 3, 1, 4).contiguous()
    patches = patches.view(patches.size(0), padded_H, padded_W)
    
    # 去掉补齐的部分
    restored_image = patches[:, :H, :W]
    
    return restored_image
