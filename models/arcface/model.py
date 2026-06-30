import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from models import BaseFaceModel


class ArcFaceModel(BaseFaceModel):
    """ArcFace face recognition model using insightface pre-trained weights.

    Uses the buffalo_l model from insightface which includes
    RetinaFace detection + ArcFace recognition (ResNet100 or MobileFaceNet).
    Handles both color (BGR) and grayscale inputs.
    """

    def __init__(self, model_name: str = 'buffalo_l'):
        self._model_name = model_name
        self._feature_dim = 512
        self._app = None
        self._load_model()

    def _load_model(self):
        import insightface
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name=self._model_name,
            providers=['CPUExecutionProvider']
        )
        self._app.prepare(ctx_id=-1, det_size=(224, 224))

    def extract_feature(self, img: np.ndarray) -> np.ndarray:
        """Extract 512-dim face embedding from a face image.

        Args:
            img: BGR image (H, W, 3) or grayscale (H, W).

        Returns:
            L2-normalized 512-dim feature vector.

        Raises:
            ValueError: If no face is detected in the image.
        """
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        faces = self._app.get(img)
        if len(faces) == 0:
            raise ValueError("No face detected in the image")

        return faces[0].embedding

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized features."""
        dot = float(np.dot(feat1, feat2))
        return float(np.clip(dot, -1.0, 1.0) * 0.5 + 0.5)

    @property
    def name(self) -> str:
        return f"ArcFace-{self._model_name}"

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def batch_extract(self, images: list) -> np.ndarray:
        """Extract features for a batch of images of the same person
        and return the mean feature.
        """
        features = []
        for img in images:
            try:
                feat = self.extract_feature(img)
                features.append(feat)
            except ValueError:
                continue
        if not features:
            raise ValueError("No face detected in any image")
        return np.mean(features, axis=0)


class ArcFaceSelfTrainedModel(BaseFaceModel):
    """Self-trained ArcFace model using OpenCV Haar Cascade detection.

    This model uses the same approach as the standalone ArcFace demo:
      - OpenCV Haar Cascade for face detection
      - Simple crop with 20% margin (no 5-point alignment)
      - ToTensor + Normalize(0.5, 0.5) → [-1, 1] preprocessing
      - IResNet-50 backbone with user-trained weights
    """

    def __init__(self, model_path: str = None):
        self._model_path = model_path or os.path.join(
            'E:', os.sep, 'AI_Curriculum_Design', 'ArcFace',
            'arcface_trained', 'work_dirs', 'arcface_trained', 'model_final.pt'
        )
        self._feature_dim = 512
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._backbone = None
        self._detector = None
        self._load_model()

    def _load_model(self):
        from .iresnet import iresnet50

        # ── detector: OpenCV Haar Cascade ──
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)

        # ── backbone: IResNet-50 with user-trained weights ──
        self._backbone = iresnet50(num_features=512)
        state_dict = torch.load(self._model_path, map_location=self._device,
                                weights_only=False)
        self._backbone.load_state_dict(state_dict, strict=True)
        self._backbone.eval()
        self._backbone = self._backbone.to(self._device)

    def _detect_and_crop(self, image_bgr: np.ndarray):
        """Detect faces using Haar Cascade and return cropped 112x112 RGB images.

        Args:
            image_bgr: BGR image (H, W, 3).

        Returns:
            List of 112x112 RGB face images.

        Raises:
            ValueError: If no face is detected.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        face_imgs = []
        for (x, y, w, h) in faces:
            # Expand crop region with 20% margin
            margin_w = int(w * 0.2)
            margin_h = int(h * 0.2)
            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(image_bgr.shape[1], x + w + margin_w)
            y2 = min(image_bgr.shape[0], y + h + margin_h)

            face_crop = image_bgr[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (112, 112))
            face_imgs.append(face_resized)

        if len(face_imgs) == 0:
            raise ValueError("No face detected in the image")
        return face_imgs

    def extract_feature(self, img: np.ndarray) -> np.ndarray:
        """Extract 512-dim face embedding using self-trained backbone.

        Args:
            img: BGR image (H, W, 3) or grayscale (H, W).

        Returns:
            L2-normalized 512-dim feature vector.

        Raises:
            ValueError: If no face is detected.
        """
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        face_imgs = self._detect_and_crop(img)
        # Use the first detected face
        face_rgb = face_imgs[0]

        # Preprocess: HWC→CHW, [0,255]→[-1,1] (matching training pipeline)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        tensor = preprocess(Image.fromarray(face_rgb)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self._backbone(tensor)

        embedding = embedding.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.astype(np.float32)

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Cosine similarity mapped to [0, 1]."""
        dot = float(np.dot(feat1, feat2))
        return float(np.clip(dot, -1.0, 1.0) * 0.5 + 0.5)

    @property
    def name(self) -> str:
        return "ArcFace(Self-Trained)"

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def batch_extract(self, images: list) -> np.ndarray:
        """Extract features for a batch and return the mean feature."""
        features = []
        for img in images:
            try:
                feat = self.extract_feature(img)
                features.append(feat)
            except ValueError:
                continue
        if not features:
            raise ValueError("No face detected in any image")
        return np.mean(features, axis=0)
