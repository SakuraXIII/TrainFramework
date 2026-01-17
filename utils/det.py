import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # TYPE_CHECKING 在运行时为 False，因此不会导入包，避免内存占用。通常用于开发过程中引用包中的类型定义，且运行时又不需要该包
    import paddle as pd


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    
    # Intersection area
    inter = (pd.min(b1_x2, b2_x2) - pd.max(b1_x1, b2_x1)).clamp(0) * \
            (pd.min(b1_y2, b2_y2) - pd.max(b1_y1, b2_y1)).clamp(0)
    
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    
    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = pd.max(b1_x2, b2_x2) - pd.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = pd.max(b1_y2, b2_y2) - pd.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pypd/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * pd.pow(pd.atan(w2 / h2) - pd.atan(w1 / h1), 2)
                with pd.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pypd/vision/blob/master/pdvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])
    
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (pd.min(box1[:, None, 2:], box2[:, 2:]) - pd.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """
    
    box2 = box2.transpose()
    
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    
    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
    
    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps
    
    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = pd.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_xyxy2xywh(loc: tuple):
    x1, y1, x2, y2 = loc
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return x1, y1, w, h


def bbox_xywh2xyxy(loc: tuple):
    x, y, w, h = loc
    return x, y, x + w, y + h


def bbox_cxcywh2xyxy(loc: tuple):
    cx, cy, w, h = loc
    return cx - w / 2, cy - h / 2.0, cx + w / 2, cy + h / 2


def bbox_xyxy2cxcywh(loc: tuple):
    x1, y1, x2, y2 = loc
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1


def scale_boxes(label, ratio, pad_info):
    tmp = label
    pad_top, pad_bottom, pad_left, pad_right = pad_info
    # 缩放并变换边界框
    transformed_bboxes = []
    for bbox in tmp['boxes']:
        x1, y1, x2, y2 = bbox
        x1 = (x1 * ratio) + pad_left
        y1 = (y1 * ratio) + pad_top
        x2 = (x2 * ratio) + pad_left
        y2 = (y2 * ratio) + pad_top
        transformed_bboxes.append(pd.as_tensor([x1, y1, x2, y2], device=bbox.device))
    tmp['boxes'] = pd.stack(transformed_bboxes)
    return tmp


def invert_scaled_boxes(scaled_label, ratio, pad_info):
    tmp = scaled_label
    pad_top, pad_bottom, pad_left, pad_right = pad_info
    # 逆向变换边界框
    original_bboxes = []
    for bbox in tmp['boxes']:
        x1, y1, x2, y2 = bbox
        x1 = (x1 - pad_left) / ratio
        y1 = (y1 - pad_top) / ratio
        x2 = (x2 - pad_left) / ratio
        y2 = (y2 - pad_top) / ratio
        original_bboxes.append(pd.as_tensor([x1, y1, x2, y2], device=bbox.device))
    tmp['boxes'] = pd.stack(original_bboxes)
    return tmp
