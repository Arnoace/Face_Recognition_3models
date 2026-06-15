# FaceR 多模型人脸识别系统模型接入说明

## 项目模型架构

系统集成了三种不同的人脸特征提取模型，并通过统一的 `ModelManager` 进行管理和调度。

| 注册名称    | 模型类              | 文件位置                          | 特征维度  |
| ------- | ---------------- | ----------------------------- | ----- |
| ArcFace | ArcFaceModel     | `models/arcface/model.py`     | 512   |
| ModelA  | FisherfacesModel | `models/model_a/interface.py` | 10000 |
| ModelB  | FaceNetModel     | `models/model_b/interface.py` | 512   |

---

# 1. 模型注册入口

## 文件位置

```text
backend/app.py
```

## 导入模型

```python
from models.arcface import ArcFaceModel
from models.model_a import FisherfacesModel
from models.model_b import FaceNetModel
```

对应关系：

```text
ArcFaceModel
└── models/arcface/model.py

FisherfacesModel
└── models/model_a/interface.py

FaceNetModel
└── models/model_b/interface.py
```

---

## 模型初始化

系统启动时创建三个模型实例：

```python
model_manager = ModelManager()

m_arcface = ArcFaceModel()
model_manager.register("ArcFace", m_arcface)

m_modela = FisherfacesModel()
model_manager.register("ModelA", m_modela)

m_modelb = FaceNetModel()
model_manager.register("ModelB", m_modelb)
```

注册完成后：

```python
model_manager._models = {
    "ArcFace": ArcFaceModel(),
    "ModelA": FisherfacesModel(),
    "ModelB": FaceNetModel()
}
```

---

## 路由注入

```python
app.include_router(
    create_model_router(model_manager)
)

app.include_router(
    create_employee_router(
        db,
        model_manager,
        feature_db
    )
)

app.include_router(
    create_recognition_router(
        model_manager,
        feature_db
    )
)
```

因此所有业务接口都通过同一个 `ModelManager` 访问模型。

---

# 2. ModelManager 调度中心

## 文件位置

```text
models/model_manager.py
```

## 核心职责

负责：

* 保存所有模型实例
* 管理当前使用模型
* 转发特征提取请求
* 提供模型切换能力

---

## 内部结构

```python
class ModelManager:

    _models = {}

    _current = "ArcFace"
```

例如：

```python
_models = {
    "ArcFace": ArcFaceModel(),
    "ModelA": FisherfacesModel(),
    "ModelB": FaceNetModel()
}
```

默认模型：

```python
_current = "ArcFace"
```

---

## 模型切换

```python
model_manager.switch(name)
```

作用：

```python
_current = name
```

例如：

```python
model_manager.switch("ModelA")
```

执行后：

```python
_current = "ModelA"
```

---

## 特征提取

统一接口：

```python
model_manager.extract_feature(img)
```

内部实现：

```python
return self.current.extract_feature(img)
```

等价于：

```python
_models[_current].extract_feature(img)
```

因此识别阶段真正执行哪个模型，由 `_current` 决定。

---

# 3. 模型切换流程

## 路由文件

```text
backend/routers/model.py
```

## 接口

```http
POST /api/model/switch?model_name=ArcFace
```

或

```http
POST /api/model/switch?model_name=ModelA
```

或

```http
POST /api/model/switch?model_name=ModelB
```

---

## 调用链

```text
POST /api/model/switch
        │
        ▼
backend/routers/model.py
        │
        ▼
model_manager.switch(model_name)
        │
        ▼
修改 _current
```

---

# 4. 人脸识别流程

## 路由文件

```text
backend/routers/recognition.py
```

## 接口

```http
POST /api/recognize
```

---

## 核心代码

```python
feat = model_manager.extract_feature(img)

return feature_db.find_match(feat)
```

---

## 调用链

```text
POST /api/recognize
        │
        ▼
recognition.py
        │
        ▼
model_manager.extract_feature(img)
        │
        ▼
当前模型.extract_feature(img)
        │
        ▼
得到特征向量
        │
        ▼
feature_db.find_match(feat)
        │
        ▼
返回识别结果
```

---

## 运行示例

当：

```python
_current = "ArcFace"
```

执行：

```python
ArcFaceModel.extract_feature(img)
```

---

当：

```python
_current = "ModelA"
```

执行：

```python
FisherfacesModel.extract_feature(img)
```

---

当：

```python
_current = "ModelB"
```

执行：

```python
FaceNetModel.extract_feature(img)
```

---

# 5. 员工注册流程

## 路由文件

```text
backend/routers/employee.py
```

## 接口

```http
POST /api/employees
```

---

## 核心代码

```python
for mn in model_manager.available_models:

    model = model_manager._models.get(mn)

    feat = model.extract_feature(img)

    all_features[mn] = feat
```

---

## 特点

注册阶段不会只提取当前模型特征。

系统会遍历所有模型：

```text
ArcFace
ModelA
ModelB
```

全部执行一次特征提取。

---

## 调用链

```text
POST /api/employees
        │
        ▼
employee.py
        │
        ▼
遍历全部模型
        │
        ├── ArcFace.extract_feature()
        ├── ModelA.extract_feature()
        └── ModelB.extract_feature()
        │
        ▼
生成 all_features
        │
        ▼
存入 feature_db
```

---

## 存储结构

```python
all_features = {
    "ArcFace": arcface_feature,
    "ModelA": fisherfaces_feature,
    "ModelB": facenet_feature
}
```

这样后续切换任意模型识别时，都能够直接使用对应模型的特征库。

---

# 6. ArcFace 模型实现

## 文件

```text
models/arcface/model.py
```

## 特征提取流程

```text
输入图像
    │
    ▼
FaceAnalysis.get(img)
    │
    ▼
人脸检测
    │
    ▼
Embedding提取
    │
    ▼
512维特征向量
```

## 输出

```python
shape = (512,)
```

---

# 7. FisherfacesModel（ModelA）

## 文件

```text
models/model_a/interface.py
```

## 特征提取流程

```text
BGR图像
    │
    ▼
转灰度
    │
    ▼
Resize(100×100)
    │
    ▼
Flatten
    │
    ▼
L2 Normalize
    │
    ▼
10000维特征
```

## 输出

```python
shape = (10000,)
```

---

# 8. FaceNetModel（ModelB）

## 文件

```text
models/model_b/interface.py
```

---

## 内部模型加载

```python
from .facenet.model import load_model
```

实际加载：

```text
models/model_b/facenet/model.py
```

---

## 权重文件

```text
models/model_b/models/facenet_model_v2.pth
```

---

## 网络结构

```text
GoogLeNet Backbone
```

---

## 特征提取流程

```text
BGR
 │
 ▼
RGB
 │
 ▼
Resize(224×224)
 │
 ▼
Normalize
 │
 ▼
GoogLeNet Forward
 │
 ▼
Embedding
 │
 ▼
L2 Normalize
 │
 ▼
512维特征
```

## 输出

```python
shape = (512,)
```

---

# 9. FaceNet 模型内部调用关系

```text
FaceNetModel
(models/model_b/interface.py)

        │
        ▼

load_model()
(models/model_b/facenet/model.py)

        │
        ▼

加载权重

models/model_b/models/
└── facenet_model_v2.pth

        │
        ▼

GoogLeNet 网络

        │
        ▼

extract_feature()
```

---

# 10. 完整调用流程图

```text
系统启动
│
├── ArcFaceModel()
│     └── models/arcface/model.py
│
├── FisherfacesModel()
│     └── models/model_a/interface.py
│
└── FaceNetModel()
      ├── models/model_b/interface.py
      └── load_model()
            └── models/model_b/facenet/model.py

                ↓

        注册到 ModelManager

                ↓

────────────────────────────────

模型切换

POST /api/model/switch
        │
        ▼
model_manager.switch()
        │
        ▼
修改 _current

────────────────────────────────

人脸识别

POST /api/recognize
        │
        ▼
model_manager.extract_feature()
        │
        ▼
当前模型.extract_feature()
        │
        ▼
feature_db.find_match()

────────────────────────────────

员工注册

POST /api/employees
        │
        ▼
遍历全部模型
        │
        ├── ArcFace.extract_feature()
        ├── ModelA.extract_feature()
        └── ModelB.extract_feature()
        │
        ▼
生成 all_features
        │
        ▼
存入 feature_db
```

---

# 11. 设计特点

## 统一接口

三个模型均实现：

```python
extract_feature(img)
```

因此业务层无需关心具体模型实现。

---

## 动态切换

通过：

```python
model_manager.switch()
```

即可在运行时切换识别模型。

---

## 多模型共存

系统启动时一次性加载：

* ArcFace
* Fisherfaces
* FaceNet

避免识别阶段重复加载模型。

---

## 多特征库存储

员工注册时同时生成：

```text
ArcFace 特征库
ModelA 特征库
ModelB 特征库
```

保证任意模型切换后均可直接识别。

---

## 低耦合设计

业务层统一调用：

```python
model_manager.extract_feature(img)
```

无需修改业务代码即可接入新的模型实现。
