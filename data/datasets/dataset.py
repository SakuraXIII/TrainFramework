import os
from functools import lru_cache
from pathlib import Path
from typing import Union, List

import cv2
import paddle as pd
import paddle.vision.transforms as T
from paddle.io import Dataset
from paddle.io.dataloader.dataset import _T

from utils.common import read_json_fromfile, init_logger

logger = init_logger()


class BaseDataset(Dataset):
    """
    root
    |---images
    |   |---image1.jpg
    |   |---image2.jpg
    |   |---...
    |
    |---annotations
    |   |---image1.json
    |   |---image2.json
    |   |---...
    |
    |---masks（仅分割任务需要）
    |   |---mask1.jpg
    |   |---mask2.jpg
    |   |---...
    |
    |---class_names.txt（可选）
    |---train.txt
    |---val.txt
    |---test.txt
    """
    
    def __init__(self, mode: str, dataset_root: Union[str, Path], transforms: List, num_classes) -> None:
        super().__init__()
        # ================ 检查必需目录 =================
        if not os.path.exists(dataset_root):
            raise FileNotFoundError("`dataset_root`路径不存在: {}".format(dataset_root))
        
        if not os.path.exists(os.path.join(dataset_root, 'images')):
            raise FileNotFoundError("图像样本需要放在 images 目录下")
        
        if not os.path.exists(os.path.join(dataset_root, 'annotations')):
            raise FileNotFoundError("标注文件需要放在 annotations 目录下")
        
        if transforms is None:
            raise ValueError("`transforms` 不能为 None")
        
        if num_classes < 1:
            raise ValueError("`num_classes` 应大于 1")
        
        # ================ 初始化变量 =================
        self.mode = mode
        self.root = Path(dataset_root)
        self.images_path = self.root / "images"
        self.annos_path = self.root / "annotations"
        self.data_list = list(str())  # 在实现类中初始化具体的数据文件
        self.transforms = T.Compose(transforms)
        self.num_classes = num_classes
        self.cls_to_id = dict()
        
        if (map_file := self.root / "class_names.txt").exists():
            self.cls_to_id = {class_name: i for i, class_name in enumerate(map_file.read_text().splitlines())}
            assert len(self.cls_to_id) == self.num_classes + 1, "`class_names.txt` 类别数与 `num_classes` 不一致"
        else:
            logger.warning("无类别ID映射文件: `class_names.txt`，无法映射类别")
        # ================ 检查索引文件 =================
        file_list = None
        match self.mode:
            case "train":
                file_list = self.root / "train.txt"
            case "val":
                file_list = self.root / "val.txt"
            case "test":
                file_list = self.root / "test.txt"
            case _:
                raise ValueError("mode 只能为 ['train', 'val', 'test'], 实际为 {}.".format(self.mode))
        
        if not file_list.exists():
            raise FileNotFoundError("未找到数据集索引文件：{}".format(file_list))
        
        self.data_list = file_list.read_text().splitlines()
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    @lru_cache
    def get_class_name(self, cls_id: int):
        return next((item[0] for item in self.cls_to_id.items() if item[1] == cls_id), -1)
    
    def get_class_id(self, cls_name: str):
        return self.cls_to_id[cls_name]
    
    @staticmethod
    def collect_fn(batch):
        raise NotImplementedError("子类需要自定义DataLoader collect_fn时，需要实现该方法")


class ClsDataset(BaseDataset):
    
    def __init__(self, mode: str, dataset_root: Union[str, Path], transforms, num_classes) -> None:
        super().__init__(mode, dataset_root, transforms, num_classes)
        self.cls_list = []
        for item in self.data_list:
            json_path = os.path.join(self.annos_path, item + '.json')
            obj = read_json_fromfile(json_path)
            if 'cls' in obj:  # 'cls': ''
                self.cls_list.append((obj['image_name'], self.get_class_id(obj['cls'])))
            else:
                raise KeyError("json标注文件未包含`cls`类别属性：{}".format(json_path))
        
        assert len(self.cls_list) == len(self.data_list), "有效标注文件数量与样本文件数量不一致"
    
    def __getitem__(self, idx: int) -> _T:
        image_name, class_id = self.cls_list[idx]
        image_path = os.path.join(self.images_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        if img is None:
            raise ValueError('无法读取图像数据，检查文件是否存在或路径包含中文: {}!'.format(image_path))
        # img = img.astype('float32')
        img = self.transforms(img)
        target = pd.tensor(class_id, dtype="int8")
        return img, target


class DetDataset(BaseDataset):
    
    def __init__(self, mode: str, dataset_root: Union[str, Path], transforms: List, num_classes) -> None:
        super().__init__(mode, dataset_root, transforms, num_classes)
        self.bbox_list = []
        for item in self.data_list:
            json_path = os.path.join(self.annos_path, item + '.json')
            obj = read_json_fromfile(json_path)
            if 'det' in obj:  # 'det': { 'bbox': [[]],'cls': [] }
                tmp = obj['det']
                assert 'cls' in tmp and 'bbox' in tmp, "det 需要包含 `bbox` 坐标属性与对应的 `cls` 类别属性"
                cbi = zip(tmp.get('cls', -1), tmp['bbox'])
                self.bbox_list.append((obj['image_name'], tuple(cbi)))
            else:
                raise KeyError("json标注文件未包含`det`坐标属性：{}".format(json_path))
        
        assert len(self.bbox_list) == len(self.data_list), "有效标注文件数量与样本文件数量不一致"
    
    def __getitem__(self, idx: int) -> _T:
        image_name, det_anno = self.bbox_list[idx]
        image_path = os.path.join(self.images_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        if img is None:
            raise ValueError('无法读取图像数据，检查文件是否存在或路径包含中文: {}!'.format(image_path))
        img = self.transforms(img)
        return img, det_anno[0], det_anno[1]
    
    @staticmethod
    def collect_fn(batch):
        return tuple(zip(*batch))


class SegDataset(BaseDataset):
    
    def __getitem__(self, idx: int) -> _T:
        pass


if __name__ == '__main__':
    # datasets = ClsDataset("train", r"", [], 2)
    dataset1 = DetDataset("train", r"", [], 2)
    print(dataset1)
    from paddle.io import DataLoader
    
    loader = DataLoader(dataset1, batch_size=8, return_list=False)
    for image, data in loader:
        print(image, data)
