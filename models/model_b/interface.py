"""FaceNet 模型接口

支持两个模型:
  1. facenet_pretrained  VGGFace2 预训练的 InceptionResNetV1 (331万张) [默认]
  2. facenet_v6          本地训练的 FaceNet (GoogLeNet, 68人)

用法:
    model = FaceNetModel(model_type="facenet_pretrained")
    feat = model.extract_feature(img)
    sim = model.compute_similarity(feat1, feat2)

本模块与 models/__init__.py 中的 BaseFaceModel 接口兼容，
可在 ModelManager 中与 ArcFace / Fisherfaces 统一注册。
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from models import BaseFaceModel

logger = logging.getLogger(__name__)

# ArcFace 标准五点对齐目标点（112×112 对齐用）
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

_BASE_DIR = Path(__file__).parent


class FaceNetModel(BaseFaceModel):
    """FaceNet 人脸识别模型

    封装 FaceNet（InceptionResNetV1 / GoogLeNet）特征提取，
    提供统一的 BaseFaceModel 接口以集成到多模型人脸识别系统。

    Args:
        model_type: "facenet_pretrained" (VGGFace2) 或 "facenet_v6" (本地训练)
        device: 推理设备，默认自动选择 CUDA 或 CPU
    """

    def __init__(self, model_type: str = "facenet_pretrained", device: str = None):
        self._model_type = model_type
        self._feature_dim = 512
        if device:
            self._device = device
        else:
            try:
                import torch
                self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self._device = 'cpu'
        self._model = None
        self._detector = None

        self._load_detector()
        self._load_model()

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _torch_available() -> bool:
        try:
            import torch  # noqa
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # 检测器加载
    # ------------------------------------------------------------------

    def _load_detector(self):
        """加载 InsightFace 人脸检测器（用于五点对齐）"""
        try:
            from insightface.app import FaceAnalysis
            self._detector = FaceAnalysis(
                name='buffalo_l',
                root=str(Path.home() / '.insightface'),
                providers=['CPUExecutionProvider']
            )
            self._detector.prepare(ctx_id=-1)
            logger.info("[FaceNet] InsightFace detector loaded")
        except Exception as e:
            logger.warning("[FaceNet] InsightFace detector not available: %s", e)

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _load_model(self):
        """根据 model_type 加载对应的 FaceNet 模型"""
        model_dir = _BASE_DIR / "models"

        if self._model_type == "facenet_pretrained":
            self._load_pretrained(model_dir)
        elif self._model_type == "facenet_v6":
            self._load_v6(model_dir)
        else:
            raise ValueError(
                f"未知模型类型: {self._model_type}，可选: facenet_pretrained, facenet_v6"
            )

    def _load_pretrained(self, model_dir: Path):
        """加载 VGGFace2 预训练 InceptionResNetV1"""
        model_path = model_dir / "facenet_pytorch_vggface2.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"[FaceNet] 预训练模型文件不存在: {model_path}\n"
                f"请将 facenet_pytorch_vggface2.pt 放入 {model_dir}"
            )

        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
        except ImportError:
            raise ImportError(
                "[FaceNet] 需要安装 PyTorch 和 facenet-pytorch:\n"
                "  pip install torch torchvision facenet-pytorch"
            )

        self._model = InceptionResnetV1(classify=False, pretrained=None).eval()
        state = torch.load(str(model_path), map_location=self._device, weights_only=True)
        self._model.load_state_dict(state, strict=False)
        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info("[FaceNet] Pretrained model loaded: %s (device=%s)", model_path, self._device)

    def _load_v6(self, model_dir: Path):
        """加载本地训练的 FaceNet (GoogLeNet)"""
        model_path = model_dir / "facenet_model_v6.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"[FaceNet] 本地训练模型文件不存在: {model_path}\n"
                f"请将 facenet_model_v6.pth 放入 {model_dir}"
            )

        # 将 model_b 目录加入 sys.path 以便导入 facenet 子包
        mb_dir = str(_BASE_DIR)
        if mb_dir not in sys.path:
            sys.path.insert(0, mb_dir)

        try:
            from facenet.model import load_model as load_facenet_v6
        except ImportError:
            raise ImportError(
                "[FaceNet] 无法导入 facenet.model，检查模型文件完整性"
            )

        self._model = load_facenet_v6(model_path, device=self._device)
        self._model.eval()
        logger.info("[FaceNet] v6 model loaded: %s (device=%s)", model_path, self._device)

    # ------------------------------------------------------------------
    # 人脸对齐
    # ------------------------------------------------------------------

    def _align_face(self, img: np.ndarray) -> np.ndarray:
        """检测人脸并进行五点仿射对齐，返回 112×112 RGB 图像

        Args:
            img: BGR 图像 (H, W, 3)

        Returns:
            对齐后的 RGB 图像 (112, 112, 3)
        """
        if self._detector is None:
            # 无检测器：中心裁剪兜底
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            s = min(h, w)
            crop = gray[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
            crop = cv2.resize(crop, (112, 112))
            return cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self._detector.get(rgb, max_num=1)

        if faces and len(faces) > 0:
            face = sorted(faces, key=lambda x: x.det_score, reverse=True)[0]
            kps = face.kps.astype(np.float32)
            M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)
            if M is None:
                M = cv2.getAffineTransform(kps[:3].astype(np.float32), ARCFACE_DST[:3].astype(np.float32))
            aligned = cv2.warpAffine(img, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)
            return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        else:
            # 无检测结果：中心裁剪兜底
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            s = min(h, w)
            crop = gray[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
            crop = cv2.resize(crop, (112, 112))
            return cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

    # ------------------------------------------------------------------
    # 特征提取（两种模型）
    # ------------------------------------------------------------------

    def _extract_pretrained(self, img: np.ndarray) -> np.ndarray:
        """使用预训练模型提取 512 维特征

        流程: 五点对齐 → 缩放到 160×160 → 归一化 [-1,1] → InceptionResNetV1 → L2 归一化
        """
        import torch
        import torch.nn.functional as F

        aligned_rgb = self._align_face(img)
        resized = cv2.resize(aligned_rgb, (160, 160))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).div(255)
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0).to(self._device)

        with torch.no_grad():
            emb = self._model(tensor)

        return F.normalize(emb, p=2, dim=1).squeeze(0).cpu().numpy()

    def _extract_v6(self, img: np.ndarray) -> np.ndarray:
        """使用本地训练模型提取 512 维特征

        流程: 五点对齐 + CLAHE+Gamma → GoogLeNet → L2 归一化
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms

        mb_dir = str(_BASE_DIR)
        if mb_dir not in sys.path:
            sys.path.insert(0, mb_dir)

        from data_preprocessing.core.processor import FaceProcessor
        processor = FaceProcessor(target_size=(112, 112))

        processed = processor.process(img)
        if processed is None:
            raise ValueError("[FaceNet] 人脸预处理失败")

        tensor = transforms.ToTensor()(Image.fromarray(processed))
        tensor = transforms.Normalize(mean=[0.5], std=[0.5])(tensor)
        tensor = tensor.unsqueeze(0).to(self._device)

        with torch.no_grad():
            emb = self._model.extract_feature(tensor)

        return F.normalize(emb, p=2, dim=1).squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # 公共接口（BaseFaceModel）
    # ------------------------------------------------------------------

    def extract_feature(self, img: np.ndarray) -> np.ndarray:
        """从图像中提取人脸特征向量

        Args:
            img: BGR 图像 (H, W, 3) 或灰度图 (H, W)

        Returns:
            512 维 L2 归一化特征向量 (np.float32)

        Raises:
            ValueError: 模型未加载或特征提取失败
        """
        if self._model is None:
            raise ValueError("[FaceNet] 模型未加载，无法提取特征")

        # 统一输入格式为 3 通道 BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        if self._model_type == "facenet_pretrained":
            feat = self._extract_pretrained(img)
        else:
            feat = self._extract_v6(img)

        return feat.astype(np.float32)

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算两个 L2 归一化特征向量的余弦相似度

        特征已 L2 归一化，直接点积即为余弦相似度。

        Returns:
            [-1, 1] 范围内的相似度，越接近 1 越相似
        """
        dot = float(np.dot(feat1, feat2))
        return float(np.clip(dot, -1.0, 1.0))

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"FaceNet-{self._model_type}"

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
