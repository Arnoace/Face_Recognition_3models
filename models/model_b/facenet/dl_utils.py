"""FaceNet 数据工具 - 复用 data_preprocessing/core 中的预处理模块"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import os

# ==================== 智能导入预处理模块 ====================
def import_preprocessing_modules():
    """动态导入 processor 和 augmentor（兼容项目结构）"""
    try:
        # 情况1：从 facenet 包内部导入
        from ..data_preprocessing.core.processor import FaceProcessor
        from ..data_preprocessing.core.augmentor import DataAugmentor
        return FaceProcessor, DataAugmentor
    except ImportError:
        try:
            # 情况2：直接从根目录导入
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from data_preprocessing.core.processor import FaceProcessor
            from data_preprocessing.core.augmentor import DataAugmentor
            return FaceProcessor, DataAugmentor
        except ImportError:
            print("[警告] 无法找到 processor.py，使用简化版预处理")
            # 回退：简单实现
            class DummyProcessor:
                def process(self, img):
                    if len(img.shape) == 3:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = img.copy()
                    return cv2.resize(gray, (112, 112))
            class DummyAugmentor:
                def augment(self, img):
                    return img
            return DummyProcessor, DummyAugmentor


FaceProcessor, DataAugmentor = import_preprocessing_modules()


class FaceDataset(Dataset):
    """人脸数据集 - 用于已预处理图片"""
    def __init__(self, root_dir: Path, target_size=(112, 112), transform=None, augment=True):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.transform = transform
        self.augmentor = DataAugmentor() if augment else None

        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            class_dir = self.root_dir / cls
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, self.class_to_idx[cls]))

        print(f"[完成] 数据集加载完成: {len(self.samples)} 张图片，{len(self.classes)} 个类别")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 已预处理图片，直接读取灰度图，跳过二次 CLAHE+Gamma
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(self.target_size, dtype=np.uint8)

        if self.augmentor:
            img = self.augmentor.augment(img)

        img_pil = Image.fromarray(img)
        tensor = self.transform(img_pil) if self.transform else transforms.ToTensor()(img_pil)
        
        return tensor, label


def get_train_transforms():
    """训练数据增强 — 强制模型学会对齐不变性"""
    return transforms.Compose([
        transforms.Resize((256, 256)),  # 先放大，让裁剪有空间
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),  # 模拟不同裁切方式
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20, fill=0),                # 模拟头部倾斜
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        # 仿射变换：模拟对齐/不对齐之间的位置差异
        transforms.RandomAffine(
            degrees=10,
            translate=(0.15, 0.15),   # 更大平移，模拟不同裁切中心
            scale=(0.85, 1.15),        # 缩放抖动，模拟不同外扩比例
            shear=5,                    # 剪切，模拟透视角度变化
            fill=0
        ),
        transforms.ToTensor(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])


def get_test_transforms():
    """测试预处理 (GoogLeNet 需要 224x224)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def preprocess_single_image(image_path: Path):
    """单张图片预处理"""
    processor = FaceProcessor(target_size=(112, 112))
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    processed = processor.process(img)
    if processed is None:
        raise ValueError("预处理失败")
    
    tensor = transforms.ToTensor()(Image.fromarray(processed))
    tensor = transforms.Normalize(mean=[0.5], std=[0.5])(tensor)
    return tensor.unsqueeze(0)


class MergedFaceDataset(Dataset):
    """合并多个 FaceDataset，标签自动偏移保持连续

    用法:
        ds = MergedFaceDataset([dir1, dir2, ...], transform=..., augment=...)
        # 返回合并后的图片和连续标签
    """
    def __init__(self, root_dirs: list, target_size=(112, 112), transform=None, augment=True):
        self.subsets = []
        self.dataset_offsets = []
        self.cumulative_sizes = []
        self.classes = []
        self.class_to_idx = {}
        self.target_size = target_size

        label_offset = 0
        for root_dir in root_dirs:
            ds = FaceDataset(root_dir, target_size=target_size, transform=transform, augment=augment)
            self.subsets.append(ds)
            self.dataset_offsets.append(label_offset)
            prev = self.cumulative_sizes[-1] if self.cumulative_sizes else 0
            self.cumulative_sizes.append(prev + len(ds))
            for cls in ds.classes:
                self.class_to_idx[cls] = label_offset + ds.class_to_idx[cls]
                self.classes.append(cls)
            label_offset += len(ds.classes)

        print(f"[完成] 合并数据集加载完成: {len(self)} 张图片，{len(self.classes)} 个类别")

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                prev = self.cumulative_sizes[i - 1] if i > 0 else 0
                img, label = self.subsets[i][idx - prev]
                return img, label + self.dataset_offsets[i]
        raise IndexError(f"索引 {idx} 超出合并数据集范围")