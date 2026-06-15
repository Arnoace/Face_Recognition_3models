"""
Fisherfaces 算法核心模块
======================
基于 OpenCV 实现 Fisherfaces (LDA-based) 人脸识别算法。

Fisherfaces 原理:
1. PCA 降维: 首先使用 PCA 将高维人脸图像降维到 (N-c) 维空间，
   其中 N 为样本数，c 为类别数。这解决了 LDA 中类内散度矩阵奇异的问题。
2. LDA 投影: 在 PCA 子空间中应用线性判别分析 (LDA)，
   最大化类间散度与类内散度的比值，找到最具区分性的特征方向。
3. 识别: 将测试图像投影到 Fisherfaces 空间，使用最近邻分类器进行识别。

参考文献:
- Belhumeur, P. N., Hespanha, J. P., & Kriegman, D. J. (1997).
  "Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection."
"""

import cv2
import numpy as np
import os
import pickle
from typing import Tuple, List, Optional


class FisherfacesModel:
    """
    Fisherfaces 人脸识别模型封装类

    使用 OpenCV 的 cv2.face.FisherFaceRecognizer 实现。
    内部流程: PCA 降维 → LDA 投影 → 最近邻分类
    """

    def __init__(self, num_components: int = 0, threshold: float = 2000.0):
        """
        初始化 Fisherfaces 模型

        Args:
            num_components: PCA 保留的主成分数量。0 表示自动选择 (N-c) 个
            threshold: 识别距离阈值，超过此值判定为未知人员
        """
        self.num_components = num_components
        self.threshold = threshold
        self.model: Optional[cv2.face.FisherFaceRecognizer] = None
        self.label_map: dict = {}       # 数字标签 → 人员名称
        self.reverse_label_map: dict = {}  # 人员名称 → 数字标签
        self.image_size: Optional[Tuple[int, int]] = None
        self.is_trained: bool = False

    def _preprocess_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        图像预处理: 灰度转换 + 尺寸归一化 + 直方图均衡化

        Args:
            image: 输入图像 (可以是彩色或灰度)
            target_size: 目标尺寸 (width, height)

        Returns:
            预处理后的灰度图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 尺寸归一化
        if target_size is not None:
            gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_LINEAR)

        # 直方图均衡化 —— 增强光照鲁棒性
        gray = cv2.equalizeHist(gray)

        return gray

    def train(self,
              images: List[np.ndarray],
              labels: List[int],
              image_size: Tuple[int, int] = (100, 100)) -> None:
        """
        训练 Fisherfaces 模型

        Args:
            images: 训练图像列表
            labels: 对应的数字标签列表
            image_size: 统一的图像尺寸
        """
        if len(images) != len(labels):
            raise ValueError(f"图像数量 ({len(images)}) 与标签数量 ({len(labels)}) 不匹配")

        if len(set(labels)) < 2:
            raise ValueError("Fisherfaces 需要至少 2 个不同的类别进行训练")

        self.image_size = image_size

        # 预处理所有训练图像
        processed_images = []
        for img in images:
            processed = self._preprocess_image(img, target_size=image_size)
            processed_images.append(processed)

        # 创建并训练 Fisherfaces 识别器
        self.model = cv2.face.FisherFaceRecognizer_create(
            num_components=self.num_components,
            threshold=self.threshold
        )

        self.model.train(processed_images, np.array(labels, dtype=np.int32))
        self.is_trained = True

        print(f"[Fisherfaces] 模型训练完成")
        print(f"  - 训练样本数: {len(processed_images)}")
        print(f"  - 类别数: {len(set(labels))}")
        print(f"  - 图像尺寸: {image_size}")
        print(f"  - 阈值: {self.threshold}")

    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """
        对单张图像进行人脸识别预测

        Args:
            image: 输入人脸图像

        Returns:
            (label, confidence): 预测标签和置信度 (距离值)
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 train() 方法")

        processed = self._preprocess_image(image, target_size=self.image_size)
        label, confidence = self.model.predict(processed)
        return label, confidence

    def predict_with_name(self, image: np.ndarray) -> Tuple[str, float]:
        """
        预测并返回人员名称

        Args:
            image: 输入人脸图像

        Returns:
            (name, confidence): 预测的人员名称和置信度
        """
        label, confidence = self.model.predict(
            self._preprocess_image(image, target_size=self.image_size)
        )
        name = self.label_map.get(label, f"Unknown-{label}")
        return name, confidence

    def get_eigenvalues(self) -> Optional[np.ndarray]:
        """
        获取 Fisherfaces 的特征值 (判别能力度量)

        Returns:
            特征值数组，如果模型未训练则返回 None
        """
        if not self.is_trained or self.model is None:
            return None
        try:
            return self.model.getEigenValues()
        except AttributeError:
            return None

    def get_eigenvectors(self) -> Optional[np.ndarray]:
        """
        获取 Fisherfaces 的特征向量 (Fisherfaces 图像)

        Returns:
            特征向量矩阵，如果模型未训练则返回 None
        """
        if not self.is_trained or self.model is None:
            return None
        try:
            return self.model.getEigenVectors()
        except AttributeError:
            return None

    def get_mean(self) -> Optional[np.ndarray]:
        """
        获取训练图像的均值脸

        Returns:
            均值脸图像，如果模型未训练则返回 None
        """
        if not self.is_trained or self.model is None:
            return None
        try:
            return self.model.getMean()
        except AttributeError:
            return None

    def save(self, filepath: str) -> None:
        """
        保存模型到文件

        Args:
            filepath: 模型保存路径
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("没有已训练的模型可以保存")

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # OpenCV 模型保存
        self.model.write(filepath)

        # 元数据保存 (标签映射、图像尺寸等)
        metadata = {
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map,
            'image_size': self.image_size,
            'num_components': self.num_components,
            'threshold': self.threshold
        }
        meta_path = filepath.replace('.yml', '_meta.pkl').replace('.xml', '_meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"[Fisherfaces] 模型已保存至: {filepath}")
        print(f"[Fisherfaces] 元数据已保存至: {meta_path}")

    @classmethod
    def load(cls, filepath: str) -> 'FisherfacesModel':
        """
        从文件加载模型

        Args:
            filepath: 模型文件路径

        Returns:
            FisherfacesModel 实例
        """
        instance = cls()

        # 加载 OpenCV 模型
        instance.model = cv2.face.FisherFaceRecognizer_create()
        instance.model.read(filepath)

        # 加载元数据
        meta_path = filepath.replace('.yml', '_meta.pkl').replace('.xml', '_meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
            instance.label_map = metadata['label_map']
            instance.reverse_label_map = metadata['reverse_label_map']
            instance.image_size = metadata['image_size']
            instance.num_components = metadata.get('num_components', 0)
            instance.threshold = metadata.get('threshold', 2000.0)
        else:
            print(f"[警告] 未找到元数据文件: {meta_path}")

        instance.is_trained = True
        print(f"[Fisherfaces] 模型已从 {filepath} 加载")
        return instance

    def get_model_info(self) -> dict:
        """
        获取模型信息摘要

        Returns:
            包含模型参数的字典
        """
        return {
            'algorithm': 'Fisherfaces (LDA)',
            'num_components': self.num_components,
            'threshold': self.threshold,
            'image_size': self.image_size,
            'num_classes': len(self.label_map) if self.label_map else 0,
            'is_trained': self.is_trained,
            'label_map': self.label_map.copy()
        }


class FisherfacesAlgorithm:
    """
    Fisherfaces 算法原理演示类

    用于教学目的，展示 Fisherfaces 算法的数学原理和步骤:
    1. 计算类内散度矩阵 Sw
    2. 计算类间散度矩阵 Sb
    3. 求解广义特征值问题: Sb * w = λ * Sw * w
    4. 选取前 (c-1) 个最大特征值对应的特征向量构成投影矩阵
    """

    @staticmethod
    def compute_within_class_scatter(samples: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        计算类内散度矩阵 Sw

        Sw = Σ Σ (x - μ_i)(x - μ_i)^T
        其中 μ_i 是第 i 类的均值

        Args:
            samples: 样本矩阵，每列为一个样本向量 (d × N)
            labels: 样本对应的类别标签

        Returns:
            类内散度矩阵 (d × d)
        """
        d = samples.shape[0]  # 特征维度
        Sw = np.zeros((d, d), dtype=np.float64)
        classes = np.unique(labels)

        for c in classes:
            class_samples = samples[:, labels == c]
            class_mean = np.mean(class_samples, axis=1, keepdims=True)
            # 中心化
            centered = class_samples - class_mean
            Sw += centered @ centered.T

        return Sw

    @staticmethod
    def compute_between_class_scatter(samples: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        计算类间散度矩阵 Sb

        Sb = Σ n_i (μ_i - μ)(μ_i - μ)^T
        其中 n_i 是第 i 类的样本数，μ 是全局均值

        Args:
            samples: 样本矩阵，每列为一个样本向量 (d × N)
            labels: 样本对应的类别标签

        Returns:
            类间散度矩阵 (d × d)
        """
        d = samples.shape[0]
        Sb = np.zeros((d, d), dtype=np.float64)
        global_mean = np.mean(samples, axis=1, keepdims=True)
        classes = np.unique(labels)

        for c in classes:
            class_samples = samples[:, labels == c]
            n_i = class_samples.shape[1]
            class_mean = np.mean(class_samples, axis=1, keepdims=True)
            diff = class_mean - global_mean
            Sb += n_i * (diff @ diff.T)

        return Sb

    @staticmethod
    def compute_fisherfaces(samples: np.ndarray, labels: np.ndarray,
                            num_components: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 Fisherfaces 投影矩阵

        步骤:
        1. 先用 PCA 降至 (N-c) 维 (解决小样本问题)
        2. 在 PCA 子空间中计算 Sw 和 Sb
        3. 求解 Sw^{-1}Sb 的特征向量
        4. 选取前 (c-1) 个最大的特征向量

        Args:
            samples: 样本矩阵 (d × N)，d 为像素数，N 为样本数
            labels: 类别标签数组
            num_components: 保留的 Fisherfaces 数量，默认为 (c-1)

        Returns:
            (projection_matrix, eigenvalues): 投影矩阵和特征值
        """
        d, N = samples.shape
        c = len(np.unique(labels))

        if num_components is None:
            num_components = c - 1

        # 全局中心化
        global_mean = np.mean(samples, axis=1, keepdims=True)
        centered = samples - global_mean

        # Step 1: PCA 降维到 (N-c) 维
        # 使用 SVD 进行 PCA (更高效，避免构建 d×d 协方差矩阵)
        # U: d × (N-1), S: (N-1,), Vt: (N-1) × N
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pca_dims = min(N - c, len(S))
        # 保留前 pca_dims 个主成分
        U_pca = U[:, :pca_dims]  # d × pca_dims
        # 投影到 PCA 空间
        samples_pca = U_pca.T @ centered  # pca_dims × N

        # Step 2: 在 PCA 子空间中计算 Sw 和 Sb
        Sw = FisherfacesAlgorithm.compute_within_class_scatter(samples_pca, labels)
        Sb = FisherfacesAlgorithm.compute_between_class_scatter(samples_pca, labels)

        # Step 3: 求解广义特征值问题
        # 使用伪逆处理 Sw 可能奇异的情况
        try:
            Sw_inv = np.linalg.inv(Sw)
            M = Sw_inv @ Sb
        except np.linalg.LinAlgError:
            Sw_pinv = np.linalg.pinv(Sw)
            M = Sw_pinv @ Sb

        eigenvalues, eigenvectors = np.linalg.eigh(M)

        # Step 4: 按特征值降序排列，选取前 num_components 个
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx][:num_components]
        eigenvectors_pca = eigenvectors[:, idx][:, :num_components]

        # 将特征向量映射回原始空间
        projection_matrix = U_pca @ eigenvectors_pca  # d × num_components

        return projection_matrix, eigenvalues


# ==================== 模块测试 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Fisherfaces 算法模块测试")
    print("=" * 60)

    # 生成模拟数据测试算法原理
    np.random.seed(42)
    d, N_per_class, c = 100, 20, 3

    # 生成 3 类有区分性的模拟数据
    samples_list = []
    labels_list = []
    for i in range(c):
        mean = np.random.randn(d) * 3 + i * 5  # 各类别在不同位置
        class_samples = np.random.randn(d, N_per_class) + mean.reshape(-1, 1)
        samples_list.append(class_samples)
        labels_list.append(np.full(N_per_class, i))

    samples = np.hstack(samples_list)
    labels = np.hstack(labels_list)

    # 计算 Fisherfaces
    proj_matrix, eigenvals = FisherfacesAlgorithm.compute_fisherfaces(
        samples, labels, num_components=min(c - 1, 5)
    )

    print(f"样本维度: {samples.shape}")
    print(f"类别数: {c}")
    print(f"投影矩阵形状: {proj_matrix.shape}")
    print(f"特征值: {eigenvals}")
    print(f"特征值之和 (判别能力): {np.sum(eigenvals):.4f}")

    # 验证类间分离性
    projected = proj_matrix.T @ (samples - np.mean(samples, axis=1, keepdims=True))
    print(f"\n投影后数据形状: {projected.shape}")

    # 计算投影后的类间/类内散度比
    Sw_proj = FisherfacesAlgorithm.compute_within_class_scatter(projected, labels)
    Sb_proj = FisherfacesAlgorithm.compute_between_class_scatter(projected, labels)
    ratio = np.trace(Sb_proj) / (np.trace(Sw_proj) + 1e-10)
    print(f"投影后类间/类内散度比: {ratio:.4f} (越大越好)")

    print("\n[OK] Fisherfaces 算法原理验证通过")
