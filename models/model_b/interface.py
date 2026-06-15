import os
import cv2
import torch
import numpy as np
from pathlib import Path
from models import BaseFaceModel
from .facenet.model import load_model


class FaceNetModel(BaseFaceModel):
    """FaceNet 人脸识别模型 — GoogLeNet (Inception) 骨干 + 512 维三元组嵌入
    
    符合原版 FaceNet 论文 (Schroff et al., CVPR 2015):
      - 骨干网络: GoogLeNet (Inception v1)
      - 嵌入维度: 512
      - 归一化: L2 归一化
    """
    def __init__(self):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None
        self._load_model()

    def _load_model(self):
        model_path = Path(os.path.join(os.path.dirname(__file__), 'models', 'facenet_model_v2.pth'))
        if not model_path.exists():
            raise FileNotFoundError(f"FaceNet 模型文件未找到: {model_path}")
        self._model = load_model(model_path, device=self._device)
        self._model.eval()

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """预处理图像: BGR→RGB → 224x224 → 归一化 → (1,3,224,224) tensor"""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)
        return tensor

    def extract_feature(self, img: np.ndarray) -> np.ndarray:
        """提取 512 维人脸特征向量
        
        Args:
            img: BGR 图像 (H, W, 3) 或灰度图 (H, W)
        Returns:
            L2 归一化的 512 维特征向量
        """
        tensor = self._preprocess(img)
        with torch.no_grad():
            embedding = self._model.forward(tensor)
        return embedding.cpu().numpy().flatten()

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算两个特征向量的余弦相似度"""
        dot = float(np.dot(feat1, feat2))
        return float(np.clip(dot, -1.0, 1.0) * 0.5 + 0.5)

    @property
    def name(self):
        return "FaceNet"

    @property
    def feature_dim(self):
        return 512

    @property
    def is_loaded(self):
        return self._model is not None
