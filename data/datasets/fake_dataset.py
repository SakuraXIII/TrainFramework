import random

import paddle as pd
from paddle.io import Dataset
from paddle.io.dataloader.dataset import _T


class FakeClsDataset(Dataset):
    """测试用的假数据集"""
    
    def __init__(self, data_sizes=10000, num_class=1000, img_size=(224, 224)):
        super().__init__()
        self.num_sizes = data_sizes
        self.num_class = num_class
        self.img_size = img_size
    
    def __len__(self):
        return self.num_sizes
    
    def __getitem__(self, index):
        # 生成随机数据和标签
        return pd.randn(3, *self.img_size), pd.randint(0, self.num_class)


class FakeDetDataset(Dataset):
    def __init__(self, data_sizes=10000, num_class=1000, img_size=(224, 224)) -> None:
        super().__init__()
        self.num_sizes = data_sizes
        self.num_class = num_class
        self.img_size = img_size
    
    def __getitem__(self, idx: int) -> _T:
        img = pd.randn(3, *self.img_size)
        count = random.randint(0, 7)
        labels = pd.randint(0, self.num_class, [count, 1])
        b = []
        for _ in range(count):
            x1, x2, y1, y2 = (random.randint(0, self.img_size[0]),
                              random.randint(0, self.img_size[0]),
                              random.randint(0, self.img_size[1]),
                              random.randint(0, self.img_size[1]),)
            b.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        bboxs = pd.to_tensor(b)
        target = {"labels": labels, "boxes": bboxs}
        return img, target
    
    def __len__(self) -> int:
        return self.num_sizes
    
    @staticmethod
    def collect_fn(batch):
        imgs = []
        target = []
        for (img, det_anno) in batch:
            imgs.append(img)
            target.append(det_anno)
        imgs = pd.stack(imgs)
        return imgs, target
