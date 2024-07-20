#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/9/7 9:41
# @Author  : SakuHx
# @File    : datasets.py
import collections
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Tuple, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
import torch.nn.functional as F

from xml.etree.ElementTree import Element as ET_Element

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

# 日志使用方法：https://zhuanlan.zhihu.com/p/166671955
logging.basicConfig(
    format="\033[1;36m \n%(pathname)s:%(lineno)d:[%(asctime)s] {%(funcName)s} (%(levelname)s): %(message)s \033[0m",
    level=logging.INFO,
    datefmt="%m/%d %H:%M:%S"
)

# 重写print，格式化输出
printf = logging.info


# ------------------------------------------------ #


class FakeDataset(Dataset):
    """测试用的假数据集"""

    def __init__(self, data_sizes=10000, num_class=1000):
        # 生成随机数据
        self.num_sizes = data_sizes
        # 生成随机标签
        self.num_class = num_class

    def __len__(self):
        return self.num_sizes

    def __getitem__(self, index):
        # 返回数据和标签
        return torch.randn(3, 256, 256), random.randint(0, self.num_class)


class VOC2012Detection(Dataset):
    """读取解析PASCAL VOC2012数据集"""

    def __init__(self, root, transform=None, target_transform=None, image_set: str = "train"):
        self.root = Path(root, "VOCdevkit", "VOC2012")
        self.img_root = Path(self.root, "JPEGImages")
        self.annotations = Path(self.root, "Annotations").iterdir()

        # read train.txt or val.txt file
        txt_path = Path(self.root, "ImageSets", "Main", image_set + '.txt')
        assert Path(txt_path).exists(), "not found {} file.".format(image_set)

        with open(txt_path) as read:
            self.xml_list = [Path(self.annotations, line.strip() + ".xml")
                             for line in read.readlines()]

        # check file
        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        for xml_path in self.xml_list:
            assert Path(xml_path).exists(), "not found '{}' file.".format(xml_path)

        # read class_indict
        json_file = Path(self.root, 'voc2012_labels.json')
        assert Path(json_file).exists(), "{} file not exist.".format(json_file)
        json_file = open(json_file, 'r')
        self.class_dict = json.load(json_file)

        self.transforms = transform
        self.target_transforms = target_transform

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        data = self.parse_voc_xml(ET_parse(self.xml_list[idx]).getroot())["annotation"]
        img_path = Path(self.img_root, data["filename"])
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(self.annotations[idx])
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(self.annotations[idx]))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            image = self.transforms(image)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return image, target

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VOCDetection.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    @staticmethod
    def collate_fn(batch):
        """DataLoader 内置的 collate_fn 处理数据会报错"""
        return tuple(zip(*batch))


class VOC2012Dataset(Dataset):
    """读取解析PASCAL VOC2012数据集"""

    def __init__(self, root, transform=None, image_set: str = "train"):
        self.root = Path(root, "VOCdevkit", "VOC2012")
        self.img_root = Path(self.root, "JPEGImages")
        self.image_files = []

        # read train.txt or val.txt file
        txt_path = Path(self.root, "ImageSets", "Main", image_set + '.txt')
        assert Path(txt_path).exists(), "not found {} file.".format(image_set)

        with open(txt_path) as read:
            self.image_files = [Path(self.img_root, line.strip() + ".jpg") for line in read.readlines()]

        self.transforms = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = Path(self.img_root, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)

        return image


class ImgDataset(Dataset):
    def __init__(self, root, transform=None, image_set: str = ""):
        self.data_dir = root
        assert (filepath := Path(self.data_dir, image_set)), "not found {} file.".format(filepath)
        self.image_files = list(Path(filepath).iterdir())
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


class ImageNet(Dataset):
    def __init__(self, root, transform=None, image_set: str = ""):
        super().__init__()
        self.data_dir = root
        self.transform = transform
        self.image_set = image_set
        self.image_files = Path(root, "Data", "CLS-LOC")
        self.image_idx = Path(root, "ImageSets", "CLS-LOC")
        with open(self.image_idx / f"{image_set}.txt", 'r') as f:
            lines = f.readlines()
            self.image_path = [line.strip() for line in lines]

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_path, label_id = self.image_path[idx].split(" ")
        image = Image.open(self.image_files / self.image_set / image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, int(label_id)


class MSCOCO(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        # COCO 图像中的每个目标的标注值都是放在单独的一个 {} 中，通过 id 索引，而不是所有目标集中在一个 {} 中
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img = coco.loadImgs(img_id)[0]
        path = img['file_name']

        if len(anns) == 0:  # 无标注数据的图像
            return None, None

        image = Image.open(Path(self.root, path)).convert("RGB")

        boxes = []
        labels = []
        for ann in anns:
            xmin, ymin, width, height = ann['bbox']  # COCO boxes 坐标格式为 xywh
            boxes.append([xmin, ymin, xmin + width, ymin + height])  # torchvision.model 中 Faster-RCNN 接受 xyxy 格式
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def custom_collect_fn(batch):
        """COCO 中存在无标注数据的图像，返回 None，这里过滤掉"""
        batch = list(filter(lambda x: x[0] is not None, batch))
        return tuple(zip(*(batch)))


def resize_and_pad(images: List[torch.Tensor], target_size: Tuple[int, int]):
    """将图像缩放并填充到指定大小，同时记录缩放比例和填充信息。

    Args:
        images: 输入图像
        target_size: 目标大小 (height, width)
    Returns:
        Tuple[List, List, List]: 处理后的图像, 缩放比例, 填充信息
    """
    resized_images = []
    ratios = []
    pad_infos = []
    for image in images:
        _, H, W = image.shape
        target_h, target_w = target_size
        # 计算缩放比例
        ratio = min(target_w / W, target_h / H)
        new_h, new_w = int(H * ratio), int(W * ratio)
        # 缩放图像
        resized_image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
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
        resized_images.append(padded_image)
        ratios.append(ratio)
        pad_infos.append(pad_info)
    return resized_images, ratios, pad_infos


def inverse_resize_and_pad(images: List[torch.Tensor], ratios: List[float], pad_infos: List[Tuple[int, int, int, int]]):
    """根据缩放比例和填充信息，逆向恢复图像到原始大小。

    Args:
        images: 输入图像
        ratios: 缩放比例
        pad_infos: 填充信息 (top, bottom, left, right)
    Returns:
        List[torch.Tensor]: 恢复后的图像
    """
    restored_images = []
    for image, ratio, pad_info in zip(images, ratios, pad_infos):
        pad_top, pad_bottom, pad_left, pad_right = pad_info
        _, padded_h, padded_w = image.shape
        # 移除填充
        unpadded_image = image[:, pad_top:padded_h - pad_bottom, pad_left:padded_w - pad_right]
        # 恢复到原始大小
        original_h, original_w = int(unpadded_image.shape[1] / ratio), int(unpadded_image.shape[2] / ratio)
        restored_image = F.interpolate(unpadded_image.unsqueeze(0), size=(original_h, original_w), mode='bilinear', align_corners=False).squeeze(0)
        restored_images.append(restored_image)
    return restored_images


def scale_boxes(boxes, ratio, pad_info):
    pad_top, pad_bottom, pad_left, pad_right = pad_info
    # 缩放并变换边界框
    transformed_bboxes = []
    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * ratio) + pad_left
        y1 = int(y1 * ratio) + pad_top
        x2 = int(x2 * ratio) + pad_left
        y2 = int(y2 * ratio) + pad_top
        transformed_bboxes.append([x1, y1, x2, y2])
    return transformed_bboxes


def invert_scaled_boxes(scaled_boxes, ratio, pad_info):
    pad_top, pad_bottom, pad_left, pad_right = pad_info
    # 逆向变换边界框
    original_bboxes = []
    for bbox in scaled_boxes:
        x1, y1, x2, y2 = bbox
        x1 = int((x1 - pad_left) / ratio)
        y1 = int((y1 - pad_top) / ratio)
        x2 = int((x2 - pad_left) / ratio)
        y2 = int((y2 - pad_top) / ratio)
        original_bboxes.append([x1, y1, x2, y2])
    return original_bboxes
