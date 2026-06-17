"""InsightFace 五点对齐 + CLAHE + Gamma

与训练数据预处理方式完全一致（检测+五点对齐+112x112）。
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional


class FaceProcessor:

    def __init__(self, target_size: Tuple[int, int] = (112, 112)):
        self.target_size = target_size

        # ---------- InsightFace 检测器（检测+五点对齐） ----------
        self.insightface_loaded = False
        self.face_handler = None
        try:
            from insightface.app import FaceAnalysis
            self.face_handler = FaceAnalysis(name='buffalo_l', root=os.path.expanduser('~/.insightface'))
            self.face_handler.prepare(ctx_id=0)
            self.insightface_loaded = True
        except Exception:
            pass

        # ---------- ArcFace 标准对齐目标点 ----------
        self.arcface_dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)

        # ---------- 光照归一化 ----------
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gamma = 0.65
        self.gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")

    def _align_face(self, img: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """五点仿射对齐"""
        M, _ = cv2.estimateAffinePartial2D(kps, self.arcface_dst, method=cv2.LMEDS)
        if M is None:
            M = cv2.getAffineTransform(kps[:3].astype(np.float32), self.arcface_dst[:3].astype(np.float32))
        aligned = cv2.warpAffine(img, M, self.target_size, borderMode=cv2.BORDER_REPLICATE)
        return aligned

    def process(self, img: np.ndarray) -> Optional[np.ndarray]:
        if img is None or img.size == 0:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        # ---- InsightFace 五点对齐 ----
        aligned = None
        if self.insightface_loaded:
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = self.face_handler.get(rgb, max_num=1)
                if faces and len(faces) > 0:
                    face = sorted(faces, key=lambda x: x.det_score, reverse=True)[0]
                    kps = face.kps.astype(np.float32)
                    aligned = self._align_face(img, kps)
                    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            except Exception:
                pass

        if aligned is None:
            # 中心裁剪兜底
            h, w = gray.shape
            s = min(h, w)
            gray = gray[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
            gray = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)

        # ---- CLAHE + Gamma ----
        smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
        enhanced = self.clahe.apply(smoothed)
        final = cv2.LUT(enhanced, self.gamma_table)

        return final

    def verify_face_present(self, img: np.ndarray) -> bool:
        if img is None or img.size == 0:
            return False
        if not self.insightface_loaded:
            return False
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_handler.get(rgb, max_num=1)
            return len(faces) > 0
        except Exception:
            return False
