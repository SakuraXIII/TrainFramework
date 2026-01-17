from pathlib import Path

import cv2
from paddle.io import Dataset


class ImageNet(Dataset):
    """读取解析ImageNet数据集用于分类"""
    
    def __init__(self, root, transform=None, image_set: str = ""):
        super().__init__()
        self.data_dir = root
        self.transform = transform
        self.image_set = image_set
        self.image_files = Path(root, "Data", "CLS-LOC")
        self.image_idx = Path(root, "Annotations", "CLS-LOC")
        with open(self.image_idx / f"{image_set}.txt", 'r') as f:
            lines = f.readlines()
            self.image_path = [line.strip() for line in lines]
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        image_path, label_id = self.image_path[idx].split(" ")
        image = cv2.imread((self.image_files / self.image_set / image_path).__str__(), cv2.IMREAD_COLOR_RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, int(label_id)
