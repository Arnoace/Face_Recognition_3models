"""人脸考勤集成模块 — 给第三方 Python 项目调用

支持两个 FaceNet 模型：
  1. facenet_v6         本地训练 (GoogLeNet, 68人)
  2. facenet_pretrained  VGGFace2 预训练 (InceptionResNetV1, 331万张) [默认]

用法:
    from attendance import AttendanceSystem

    # 用预训练模型 (默认)
    system = AttendanceSystem(model_type="facenet_pretrained", threshold=0.50)

    # 用本地训练模型
    system = AttendanceSystem(model_type="facenet_v6", threshold=0.55)

    # 录入
    system.register("张三", [cv2.imread("photo1.jpg"), cv2.imread("photo2.jpg")])

    # 识别
    result = system.recognize(cv2.imread("test.jpg"))
    print(result)

    # 管理
    system.save("gallery.json")
    system.load("gallery.json")
    system.list()
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class AttendanceSystem:
    """人脸考勤系统

    Args:
        model_type: "facenet_v6" (本地训练) 或 "facenet_pretrained" (VGGFace2 预训练)
        threshold: 识别阈值
    """

    def __init__(self, model_type: str = "facenet_pretrained", threshold: float = 0.50):
        self.threshold = threshold
        self.model_type = model_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gallery = []
        self.base_dir = Path(__file__).parent

        # ArcFace 对齐点
        self.arcface_dst = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014],
            [56.0252, 71.7366], [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)

        # InsightFace 检测器
        from insightface.app import FaceAnalysis
        self.detector = FaceAnalysis(name='buffalo_l', root=str(Path.home() / '.insightface'))
        self.detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

        # 加载模型
        self._load_model()
        print(f"[考勤] 初始化完成 (model={model_type}, device={self.device}, threshold={threshold})")

    def _load_model(self):
        if self.model_type == "facenet_v6":
            from facenet.model import load_model
            path = self.base_dir / "models" / "facenet_model_v6.pth"
            self.model = load_model(path, device=self.device)
            self.model.eval()

            self._extract_embedding = self._extract_v6

        elif self.model_type == "facenet_pretrained":
            from facenet_pytorch import InceptionResnetV1
            path = self.base_dir / "models" / "facenet_pytorch_vggface2.pt"
            self.model = InceptionResnetV1(classify=False, pretrained=None).eval()
            self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True), strict=False)
            self.model = self.model.to(self.device)

            self._extract_embedding = self._extract_pretrained

    def _align_face(self, img: np.ndarray) -> np.ndarray:
        """五点对齐返回 112x112 RGB"""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector.get(rgb, max_num=1)
        if faces and len(faces) > 0:
            kps = faces[0].kps.astype(np.float32)
            M, _ = cv2.estimateAffinePartial2D(kps, self.arcface_dst, method=cv2.LMEDS)
            if M is None:
                M = cv2.getAffineTransform(kps[:3], self.arcface_dst[:3])
            aligned = cv2.warpAffine(img, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)
            return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            s = min(h, w)
            crop = gray[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
            crop = cv2.resize(crop, (112, 112))
            return cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

    def _extract_v6(self, img: np.ndarray) -> torch.Tensor:
        """本地 FaceNet v6: processor + 模型"""
        from facenet.dl_utils import import_preprocessing_modules
        FaceProcessor, _ = import_preprocessing_modules()
        processor = FaceProcessor(target_size=(112, 112))
        processed = processor.process(img)
        if processed is None:
            raise ValueError("人脸预处理失败")

        from PIL import Image
        from torchvision import transforms
        tensor = transforms.ToTensor()(Image.fromarray(processed))
        tensor = transforms.Normalize(mean=[0.5], std=[0.5])(tensor)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.extract_feature(tensor)
        return F.normalize(emb, p=2, dim=1).squeeze(0).cpu()

    def _extract_pretrained(self, img: np.ndarray) -> torch.Tensor:
        """预训练 FaceNet: 对齐 → 160×160 → 模型"""
        aligned_rgb = self._align_face(img)
        resized = cv2.resize(aligned_rgb, (160, 160))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).div(255)
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(tensor)
        return F.normalize(emb, p=2, dim=1).squeeze(0).cpu()

    def _extract_multi(self, images: list) -> list:
        return [self._extract_embedding(img) for img in images]

    # ---------- 管理 ----------

    def register(self, name: str, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        self.remove(name)
        embs = self._extract_multi(images)
        self.gallery.append({'name': name, 'embeddings': embs})
        print(f"[考勤] 录入: {name} ({len(embs)}张)")

    def remove(self, name: str):
        self.gallery = [p for p in self.gallery if p['name'] != name]

    def clear(self):
        self.gallery = []

    def list(self) -> list:
        return [p['name'] for p in self.gallery]

    # ---------- 识别 ----------

    def recognize(self, image: np.ndarray) -> dict:
        if not self.gallery:
            return {"name": None, "similarity": 0, "identified": False, "top5": []}

        probe = self._extract_embedding(image)

        results = []
        for person in self.gallery:
            person_embs = torch.stack(person['embeddings'])
            person_embs = F.normalize(person_embs, p=2, dim=1)
            best = torch.mm(probe.unsqueeze(0), person_embs.t()).max().item()
            results.append({"name": person['name'], "similarity": round(best, 4)})

        results.sort(key=lambda x: x['similarity'], reverse=True)
        top = results[0] if results else None
        return {
            "name": top['name'] if top and top['similarity'] >= self.threshold else None,
            "similarity": top['similarity'] if top else 0,
            "identified": top is not None and top['similarity'] >= self.threshold,
            "top5": results[:5],
        }

    # ---------- 持久化 ----------

    def save(self, path: str):
        data = [{'name': p['name'], 'embeddings': [e.tolist() for e in p['embeddings']]} for p in self.gallery]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[考勤] 已保存: {path} ({len(data)}人)")

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.gallery = [{'name': item['name'], 'embeddings': [torch.tensor(e) for e in item['embeddings']]} for item in data]
        print(f"[考勤] 已加载: {path} ({len(self.gallery)}人)")
