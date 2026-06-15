import cv2
import numpy as np
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
