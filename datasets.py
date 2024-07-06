#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/9/7 9:41
# @Author  : SakuHx
# @File    : datasets.py
import collections
import json
import logging
from pathlib import Path
from typing import Any, Dict
from xml.etree.ElementTree import Element as ET_Element

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection

# 日志使用方法：https://zhuanlan.zhihu.com/p/166671955
logging.basicConfig(
    format="\033[1;36m \n%(pathname)s:%(lineno)d:[%(asctime)s] {%(funcName)s} (%(levelname)s): %(message)s \033[0m",
    level=logging.INFO,
    datefmt="%m/%d %H:%M:%S"
)

# 重写print，格式化输出
printf = logging.info


# ------------------------------------------------ #


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
        self.img_files = Path(root, "Data", "CLS-LOC")
        self.img_idx = Path(root, "ImageSets", "CLS-LOC")
        with open(self.img_idx / f"{image_set}.txt", 'r') as f:
            self.img_path = f.readlines()

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path, label_id = self.img_path[idx].split(" ")
        image = Image.open(self.img_files / img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_id
