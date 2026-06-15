import os
from models import BaseFaceModel
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class FisherfacesModel(BaseFaceModel):
    def __init__(self):
        self._img_size = (100, 100)
    def extract_feature(self, img):
        import cv2
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        g = cv2.resize(g, self._img_size)
        f = g.flatten().astype('float32')
        n = __import__('numpy').linalg.norm(f); return f/n if n>0 else f
    def compute_similarity(self, a, b):
        import numpy as np
        d = float(np.dot(a, b)); return float(np.clip(d, -1, 1)*0.5+0.5)
    @property
    def name(self): return "Fisherfaces"
    @property
    def feature_dim(self): return 10000
    @property
    def is_loaded(self): return True
