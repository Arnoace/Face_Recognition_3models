"""Fisherfaces 模型接口
使用预训练的 Fisherfaces (PCA+LDA) 投影矩阵提取判别性特征。
新员工无需重训——直接投影即可获得 27 维特征向量。
"""

import os
import shutil
import tempfile
import numpy as np
import cv2
from models import BaseFaceModel

_BASE = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BASE, 'models')
_MODEL_PATH = os.path.join(_MODEL_DIR, 'fisherfaces_model.yml')


def _ascii_path(path: str) -> str:
    """将含中文的路径复制到纯 ASCII 临时路径，
    解决 OpenCV C++ 底层在 Windows 上不支持 UTF-8 路径的问题。"""
    if not os.path.exists(path):
        return path
    try:
        path.encode('ascii')
        return path
    except UnicodeEncodeError:
        pass
    # 用 SYSTEMROOT (C:\Windows) 下的临时目录，保证纯 ASCII
    sysroot = os.environ.get('SYSTEMROOT', 'C:\\Windows')
    tmp_dir = os.path.join(sysroot, 'Temp')
    os.makedirs(tmp_dir, exist_ok=True)
    ext = os.path.splitext(path)[1]
    dst = os.path.join(tmp_dir, f'_{os.urandom(4).hex()}{ext}')
    shutil.copy2(path, dst)
    return dst


class FisherfacesModel(BaseFaceModel):
    def __init__(self):
        self._eigenvectors = None   # shape (2500, 27) — Fisherfaces 投影矩阵
        self._mean_face = None      # shape (2500,) — 均值脸
        self._img_size = (50, 50)   # 模型训练时的尺寸
        self._loaded = False
        self._load_model()

    def _load_model(self):
        """加载预训练的 Fisherfaces 模型，提取投影矩阵"""
        if not os.path.exists(_MODEL_PATH):
            print(f"[Fisherfaces] 未找到预训练模型 {_MODEL_PATH}，回退到原始像素特征")
            return
        try:
            model = cv2.face.FisherFaceRecognizer_create()
            model.read(_ascii_path(_MODEL_PATH))  # 使用短路径解决中文路径问题
            self._eigenvectors = model.getEigenVectors()      # (2500, 27)
            self._mean_face = model.getMean().flatten()        # (2500,)
            self._loaded = True
            print(f"[Fisherfaces] 预训练模型已加载，特征维度: {self._eigenvectors.shape[1]}")
        except Exception as e:
            print(f"[Fisherfaces] 加载模型失败: {e}，回退到原始像素特征")

    def _preprocess(self, img):
        """预处理：灰度化 → 缩放 50×50 → 直方图均衡化 → 展平"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        gray = cv2.resize(gray, self._img_size, interpolation=cv2.INTER_LINEAR)
        gray = cv2.equalizeHist(gray)
        return gray.flatten().astype(np.float32)  # (2500,)

    def extract_feature(self, img):
        """提取 Fisherfaces 判别性特征"""
        if not self._loaded:
            # 回退：原始像素特征 (100×100 = 10000维)
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            g = cv2.resize(g, (100, 100))
            f = g.flatten().astype(np.float32)
        else:
            vec = self._preprocess(img)                     # (2500,)
            vec = vec - self._mean_face                      # 减去均值脸
            f = self._eigenvectors.T @ vec                   # 投影 → (27,)
        n = np.linalg.norm(f)
        return f / n if n > 0 else f

    def compute_similarity(self, a, b):
        """余弦相似度（特征已 L2 归一化，直接点积）"""
        return float(np.dot(a, b))

    @property
    def name(self):
        return "Fisherfaces"

    @property
    def feature_dim(self):
        return 27 if self._loaded else 10000

    @property
    def is_loaded(self):
        return True
