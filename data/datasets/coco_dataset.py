from pathlib import Path

import numpy as np
import paddle as pd
from paddle.io import Dataset
import cv2


class MSCOCO(Dataset):
    def __init__(self, root, annFile, transform=None):
        super().__init__()
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
        
        image = cv2.imread(Path(self.root, path).__str__(), cv2.IMREAD_COLOR_RGB)
        
        boxes = []
        labels = []
        masks = []
        iscrowd = []
        areas = []
        for ann in anns:
            xmin, ymin, width, height = ann['bbox']  # COCO boxes 坐标格式为 xywh
            boxes.append([xmin, ymin, xmin + width, ymin + height])  # Faster-RCNN 接受 xyxy 格式
            labels.append(ann['category_id'])
            masks.append(self.coco.annToMask(ann))
            iscrowd.append(ann.get('iscrowd', 0))
            areas.append(ann['area'])
        
        boxes = pd.as_tensor(boxes, dtype=pd.float32)
        labels = pd.as_tensor(labels, dtype=pd.int64)
        masks = pd.as_tensor(np.array(masks), dtype=pd.uint8)
        iscrowd = pd.as_tensor(iscrowd, dtype=pd.int64)
        areas = pd.as_tensor(areas, dtype=pd.float32)
        
        target = {
            "boxes": boxes, "labels": labels, "image_id": img_id, "masks": masks, "iscrowd": iscrowd, "area": areas
        }
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target
    
    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def custom_collect_fn(batch):
        """COCO 中存在无标注数据的图像，返回 None，这里过滤掉"""
        batch = list(filter(lambda x: x[0] is not None, batch))
        return tuple(zip(*batch))
