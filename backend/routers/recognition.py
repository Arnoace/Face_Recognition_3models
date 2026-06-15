import json, os, logging
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend import config
logger = logging.getLogger(__name__)
def _norm(f):
    n = np.linalg.norm(f, axis=1, keepdims=True)
    n[n==0] = 1; return f / n
class FeatureDatabase:
    def __init__(self, db=None, model_manager=None):
        self._stores = {}       # model_name → {"features": ndarray, "metadata": [], "yale": [], "employee": []}
        self._model_manager = model_manager
        self._db = db
        if db and model_manager:
            self._load_default(db, model_manager)

    def _store(self, model_name=None):
        """Get or create the feature store for a given model."""
        mn = model_name or (self._model_manager.current_name if self._model_manager else "default")
        if mn not in self._stores:
            self._stores[mn] = {
                "features": np.empty((0, 512), dtype=np.float32),
                "metadata": [],
                "yale_metadata": [],
                "db_metadata": [],
            }
        return self._stores[mn]

    @property
    def features(self):
        return self._store()["features"]
    @features.setter
    def features(self, v):
        self._store()["features"] = v
    @property
    def metadata(self):
        return self._store()["metadata"]
    @metadata.setter
    def metadata(self, v):
        self._store()["metadata"] = v
    @property
    def yale_metadata(self):
        return self._store()["yale_metadata"]
    @yale_metadata.setter
    def yale_metadata(self, v):
        self._store()["yale_metadata"] = v
    @property
    def db_metadata(self):
        return self._store()["db_metadata"]
    @db_metadata.setter
    def db_metadata(self, v):
        self._store()["db_metadata"] = v

    def _load_default(self, db, model_manager):
        """Load Yale + employee features using ALL registered models."""
        for name in model_manager.available_models:
            try:
                m = model_manager._models.get(name)
                if not m:
                    continue
                dim = m.feature_dim if hasattr(m, 'feature_dim') else 512
                store = self._stores.setdefault(name, {
                    "features": np.empty((0, dim), dtype=np.float32),
                    "metadata": [],
                    "yale_metadata": [],
                    "db_metadata": [],
                })
                # Try loading Yale features for this model
                self._load_yale_for_model(name, m, store)
                # Try loading employee features for this model
                self._load_employees_for_model(name, m, store, db)
            except Exception as e:
                logger.warning(f"Could not load features for model '{name}': {e}")

    def _load_yale_for_model(self, model_name, model, store):
        """Load pre-extracted features only for the model they were created for."""
        # Only ArcFace has pre-extracted .npy features
        p = config.ARC_FEATURES_PATH
        if not os.path.exists(p):
            return
        if model_name == "ArcFace":
            feats = np.load(p).astype(np.float32)
            labels = np.load(config.ARC_LABELS_PATH)
            with open(config.ARC_NAMES_PATH) as f: names = json.load(f)
            feats = _norm(feats)
            for i in range(len(feats)):
                name = names.get(str(int(labels[i])), f"yale_{i}")
                store["yale_metadata"].append({'id':-i-1,'name':name,'employee_id':name,'department':'YaleB','source':'yale'})
            if store["features"].size:
                store["features"] = np.vstack([store["features"], feats])
            else:
                store["features"] = feats
            store["metadata"].extend(store["yale_metadata"])
            logger.info(f"Loaded {len(feats)} Yale features for '{model_name}'")
        else:
            logger.info(f"No pre-extracted Yale features for '{model_name}', skipping Yale data")

    def _load_employees_for_model(self, model_name, model, store, db):
        """Load employee features. For non-ArcFace models, re-extract from stored images."""
        # For now, employee features are only available for ArcFace (512-dim)
        # Non-ArcFace models will work once employees are registered under that model
        feats, meta = db.get_all_features(model_name=model_name)
        if not len(feats):
            return
        # The stored features in DB are 512-dim (extracted by ArcFace at registration time)
        # Only use them if dims match
        expected_dim = model.feature_dim if hasattr(model, 'feature_dim') else 512
        if feats.shape[1] != expected_dim:
            logger.info(f"Employee features dim ({feats.shape[1]}) != model '{model_name}' dim ({expected_dim}), skipping DB features")
            return
        feats = _norm(feats.astype(np.float32))
        store["db_metadata"] = meta
        if store["features"].size:
            store["features"] = np.vstack([store["features"], feats])
        else:
            store["features"] = feats
        for m in meta: m['source'] = 'employee'
        store["metadata"].extend(meta)
        logger.info(f"Loaded {len(meta)} employee features for '{model_name}'")

    # Legacy methods kept for backward compatibility
    def _load_yale(self):
        p = config.ARC_FEATURES_PATH
        if not os.path.exists(p): return
        feats = np.load(p).astype(np.float32)
        labels = np.load(config.ARC_LABELS_PATH)
        with open(config.ARC_NAMES_PATH) as f: names = json.load(f)
        feats = _norm(feats)
        for i in range(len(feats)):
            name = names.get(str(int(labels[i])), f"yale_{i}")
            s = self._store()
            s["yale_metadata"].append({'id':-i-1,'name':name,'employee_id':name,'department':'YaleB','source':'yale'})
        if self.features.size: self.features = np.vstack([self.features, feats])
        else: self.features = feats
        s = self._store()
        s["metadata"].extend(s["yale_metadata"])
    def _load_employees(self, db):
        feats, meta = db.get_all_features()
        if not len(feats): return
        feats = _norm(feats.astype(np.float32))
        s = self._store()
        s["db_metadata"] = meta
        if self.features.size: self.features = np.vstack([self.features, feats])
        else: self.features = feats
        for m in meta: m['source'] = 'employee'
        s["metadata"].extend(meta)
    def add_employee_feature(self, pk, name, eid, dept, feat, model_name=None):
        s = self._store(model_name)
        feat = feat.astype(np.float32)
        feat /= np.linalg.norm(feat)
        if s["features"].size == 0: s["features"] = feat.reshape(1,-1)
        else: s["features"] = np.vstack([s["features"], feat.reshape(1,-1)])
        m = {'id':pk,'name':name,'employee_id':eid,'department':dept,'source':'employee'}
        s["metadata"].append(m)
        s["db_metadata"].append(m)

    def remove_employee_feature(self, pk):
        """Remove employee feature from ALL model stores."""
        for name, s in self._stores.items():
            idxs = [i for i,m in enumerate(s["metadata"]) if m['id']==pk]
            if not idxs: continue
            s["metadata"].pop(idxs[0])
            s["features"] = np.delete(s["features"], idxs[0], axis=0)
            s["db_metadata"] = [m for m in s["db_metadata"] if m['id']!=pk]

    # Legacy single-store remove kept for compat
    def _remove_employee_feature_legacy(self, pk):
        """Old single-store removal (still works for current model)."""
        s = self._store()
        idxs = [i for i,m in enumerate(s["metadata"]) if m['id']==pk]
        if not idxs: return
        s["metadata"].pop(idxs[0])
        self.features = np.delete(self.features, idxs[0], axis=0)
        s["db_metadata"] = [m for m in s["db_metadata"] if m['id']!=pk]

    def find_match(self, qf, th=None):
        if th is None: th = config.SIMILARITY_THRESHOLD
        if self.features.size == 0: return {'found':False,'similarity':0,'employee':None}
        qf = qf.astype(np.float32)
        n = np.linalg.norm(qf)
        if n > 0: qf /= n
        sims = np.dot(self.features, qf)
        bi = int(np.argmax(sims)); bs = float(sims[bi])
        emp = self.metadata[bi] if bs >= th else None
        return {'found':bs>=th,'similarity':round(bs,4),'employee':emp}
def create_recognition_router(model_manager, feature_db):
    router = APIRouter(prefix="/api/recognize", tags=["Recognition"])
    @router.post("")
    async def recognize_face(file: UploadFile = File(...)):
        c = await file.read()
        import cv2
        arr = np.frombuffer(c, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        try:
            feat = model_manager.extract_feature(img)
        except Exception as e:
            return {"code": 400, "data": {"found": False, "message": f"特征提取失败: {str(e)}"}}
        return {"code": 200, "data": feature_db.find_match(feat)}
    return router
