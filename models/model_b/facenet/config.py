from pathlib import Path

class Config:
    DATA_DIR = Path("./datasets/YaleB_processed_112")
    MODEL_DIR = Path("models")
    MODEL_TYPE = "fisher"        # 可选: "fisher", "lbph", "eigen"
    IMG_SIZE = (112, 112)
    TRAIN_RATIO = 0.8
    N_COMPONENTS = 200           # 仅 Eigenfaces 使用
    K = 5