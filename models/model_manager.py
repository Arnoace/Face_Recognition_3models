class ModelManager:
    def __init__(self):
        self._models = {}
        self._current = None
    def register(self, name, model):
        self._models[name] = model
        if self._current is None:
            self._current = name
    def switch(self, name):
        if name not in self._models:
            raise ValueError("未知模型")
        self._current = name
    @property
    def current(self):
        return self._models.get(self._current)
    @property
    def current_name(self):
        return self._current
    @property
    def available_models(self):
        return list(self._models.keys())
    def extract_feature(self, img):
        return self.current.extract_feature(img)
    def compute_similarity(self, a, b):
        return self.current.compute_similarity(a, b)
