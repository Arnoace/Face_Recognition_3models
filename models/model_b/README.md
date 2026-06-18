# 人脸考勤系统
基于 **FaceNet**（Google CVPR 2015）的人脸识别考勤系统，内含两个模型。
当前 ModelB = FaceNet，具体是 facenet_pretrained 模式：
骨干网络: InceptionResNetV1（VGGFace2 预训练，331万张人脸）
输入: 160×160 RGB，InsightFace 五点对齐
特征维度: 512
模型文件: models/model_b/models/facenet_pytorch_vggface2.pt
这是默认配置（backend/app.py:51），因为 FaceNetModel() 没有传参数，走的就是 model_type="facenet_pretrained"。
如果想切换成本地训练的版本，把 backend/app.py:51 改成：
m_modelb = FaceNetModel(model_type="facenet_v6")
## 目录结构

```
attendance_system/
├── app.py                     # Web 服务端（Flask）
├── attendance.py              # Python 集成模块
├── requirements.txt           # 依赖清单
├── README.md                  # 本文件
├── templates/
│   └── index.html             # 前端页面
├── facenet/                   # 本地 FaceNet 代码
│   ├── __init__.py
│   ├── model.py               # 模型定义 (GoogLeNet + 512维嵌入)
│   └── dl_utils.py            # 预处理工具
├── data_preprocessing/core/
│   └── processor.py           # CLAHE+Gamma 光照归一化
├── models/
│   ├── facenet_model_v6.pth            # 本地训练 (GoogLeNet, 68人)
│   └── facenet_pytorch_vggface2.pt     # VGGFace2 预训练 (InceptionResNetV1, 331万张)
├── gallery/                   # 录入照片（自动创建）
└── gallery_db.json            # 人脸库（自动创建）
```

## 两个模型对比

| 项目 | facenet_v6（本地训练） | facenet_pretrained（预训练） |
|------|:---:|:---:|
| 骨干网络 | **GoogLeNet** (Inception v1) | **InceptionResNetV1** (NN3) |
| 训练数据 | YaleB + ORL + 自拍 (68人)     | **VGGFace2** (9131人, 331万张) |
| 预处理 | CLAHE+Gamma 灰度图              | 五点对齐 RGB |
| 输入尺寸 | 112×112 灰度                  | 160×160 RGB |
| 是否需要训练 | ✅ 已训练完成              | ✅ 预训练完成，无需训练 |
| 熟人相似度 | 60-85%                      | **90-98%** |
| 陌生人拒识 | 需精细调阈值                 | 天然低分，好调 |
| 切换方式 | `MODEL_TYPE = "facenet_v6"`  | `MODEL_TYPE = "facenet_pretrained"` [默认] |

## 快速使用（Web 界面）

```powershell
pip install -r requirements.txt
python app.py
```

浏览器打开 http://localhost:5000

### 切换模型

打开 [app.py](app.py)，改第 24 行：

```python
MODEL_TYPE = "facenet_v6"            # 本地训练
MODEL_TYPE = "facenet_pretrained"     # VGGFace2 预训练 [默认]
```

## 集成到其他项目

```python
from attendance import AttendanceSystem
import cv2

# 预训练模型（推荐）
system = AttendanceSystem(model_type="facenet_pretrained", threshold=0.50)

# 或本地训练模型
# system = AttendanceSystem(model_type="facenet_v6", threshold=0.55)

# 录入
system.register("张三", [cv2.imread("photo1.jpg"), cv2.imread("photo2.jpg")])

# 识别
result = system.recognize(cv2.imread("test.jpg"))
print(result)

# 管理
system.save("gallery.json")
system.load("gallery.json")
system.list()
```

## 技术栈

- 人脸检测+对齐: InsightFace (RetinaFace + 五点对齐)
- 特征提取: **FaceNet** (InceptionResNetV1 + VGGFace2 预训练 或 GoogLeNet + 本地训练)
- 相似度: 余弦相似度
