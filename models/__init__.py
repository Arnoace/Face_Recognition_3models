from abc import ABC, abstractmethod
import numpy as np


class BaseFaceModel(ABC):
    """Abstract base interface for all face recognition models.

    All models (ArcFace, ModelA, ModelB) must implement this interface
    to ensure interchangeability in the recognition pipeline.
    """

    @abstractmethod
    def extract_feature(self, img: np.ndarray) -> np.ndarray:
        """Extract face feature embedding from an image.

        Args:
            img: BGR image as numpy array (H, W, 3).
                 Grayscale (H, W) is also accepted and will be converted.

        Returns:
            Normalized feature vector (512-dim by convention).
        """
        pass

    @abstractmethod
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute similarity score between two feature vectors.

        Args:
            feat1: First feature vector.
            feat2: Second feature vector.

        Returns:
            Similarity score in [0, 1], higher = more similar.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Output feature dimension (expected 512)."""
        pass
