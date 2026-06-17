"""人脸考勤系统 — 后端服务

支持双模型:
  1. model=facenet_v6   本地训练的 FaceNet (GoogLeNet, 68人)
  2. model=facenet_pretrained  VGGFace2 预训练 FaceNet (InceptionResNetV1)
"""

import json
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ========== 配置 ==========
BASE_DIR = Path(__file__).parent
GALLERY_DIR = BASE_DIR / "gallery"
GALLERY_DB_PATH = BASE_DIR / "gallery_db.json"
# 可选: "facenet_v6" 或 "facenet_pretrained"
MODEL_TYPE = "facenet_pretrained"
THRESHOLD = 0.50
TOP_K = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GALLERY_DIR.mkdir(exist_ok=True)

# ========== 两个模型的预处理对齐点 ==========
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# ========== InsightFace 检测器（两个模型共用） ==========
from insightface.app import FaceAnalysis
face_detector = FaceAnalysis(name='buffalo_l', root=str(Path.home() / '.insightface'))
face_detector.prepare(ctx_id=0)


def align_face(img: np.ndarray) -> np.ndarray:
    """检测人脸 + 五点对齐，返回 112x112 RGB"""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_detector.get(rgb, max_num=1)
    if faces and len(faces) > 0:
        face = sorted(faces, key=lambda x: x.det_score, reverse=True)[0]
        kps = face.kps.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)
        if M is None:
            M = cv2.getAffineTransform(kps[:3].astype(np.float32), ARCFACE_DST[:3].astype(np.float32))
        aligned = cv2.warpAffine(img, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)
        return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        s = min(h, w)
        crop = gray[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
        crop = cv2.resize(crop, (112, 112))
        return cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)


# ========== 加载模型 ==========
print(f"[考勤系统] 模型类型: {MODEL_TYPE}, 设备: {device}")

if MODEL_TYPE == "facenet_v6":
    # ---- 本地训练的 FaceNet (GoogLeNet + CLAHE+Gamma) ----
    from facenet.model import load_model
    MODEL_PATH = BASE_DIR / "models" / "facenet_model_v6.pth"
    model = load_model(MODEL_PATH, device=device)
    model.eval()

    def extract_embedding(image_data: bytes) -> torch.Tensor:
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("图片解码失败")

        # align_face 已在 processor.py 中做了五点对齐+CLAHE+Gamma
        from facenet.dl_utils import import_preprocessing_modules
        FaceProcessor, _ = import_preprocessing_modules()
        processor = FaceProcessor(target_size=(112, 112))
        processed = processor.process(img)
        if processed is None:
            raise ValueError("人脸预处理失败")

        from PIL import Image
        from torchvision import transforms
        tensor = transforms.ToTensor()(Image.fromarray(processed))
        tensor = transforms.Normalize(mean=[0.5], std=[0.5])(tensor)
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.extract_feature(tensor)
        return F.normalize(emb, p=2, dim=1).cpu()

elif MODEL_TYPE == "facenet_pretrained":
    # ---- VGGFace2 预训练 FaceNet (InceptionResNetV1, 331万张) ----
    from facenet_pytorch import InceptionResnetV1
    MODEL_PATH = BASE_DIR / "models" / "facenet_pytorch_vggface2.pt"
    model = InceptionResnetV1(classify=False, pretrained=None).eval()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    model = model.to(device)

    def extract_embedding(image_data: bytes) -> torch.Tensor:
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("图片解码失败")

        # 五点对齐 → 160×160 RGB → 模型
        aligned_rgb = align_face(img)
        resized = cv2.resize(aligned_rgb, (160, 160))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).div(255)
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(tensor)
        return F.normalize(emb, p=2, dim=1).cpu()

print(f"[考勤系统] 模型加载完成")


# ========== 人脸库 ==========
gallery_db = []
if GALLERY_DB_PATH.exists():
    with open(GALLERY_DB_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            item['embeddings'] = [torch.tensor(e) for e in item['embeddings']]
            gallery_db.append(item)
    print(f"[考勤系统] 已加载 {len(gallery_db)} 个已注册人员")


def save_gallery_db():
    data = []
    for item in gallery_db:
        data.append({
            'name': item['name'],
            'image': item['image'],
            'embeddings': [e.tolist() for e in item['embeddings']],
        })
    with open(GALLERY_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ========== 页面 ==========
@app.route('/')
def index():
    return render_template('index.html')


# ========== API: 录入 ==========
@app.route('/api/register', methods=['POST'])
def register():
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'message': '请输入姓名'})
    if any(item['name'] == name for item in gallery_db):
        return jsonify({'success': False, 'message': f'姓名 "{name}" 已存在'})

    embs = []
    saved_count = 0
    for i in range(10):
        key = f'image{i}'
        if key not in request.files:
            break
        img_bytes = request.files[key].read()
        if not img_bytes:
            continue
        try:
            emb = extract_embedding(img_bytes)
            embs.append(emb)
        except Exception:
            continue
        if saved_count == 0:
            person_dir = GALLERY_DIR / name
            person_dir.mkdir(exist_ok=True)
            filename = f"{uuid.uuid4().hex[:8]}.jpg"
            (person_dir / filename).write_bytes(img_bytes)
            image_ref = f"{name}/{filename}"
            saved_count += 1

    if not embs:
        return jsonify({'success': False, 'message': '未接收到有效图片'})

    gallery_db.append({
        'name': name,
        'image': image_ref,
        'embeddings': [e.squeeze(0) for e in embs],
    })
    save_gallery_db()
    return jsonify({'success': True, 'message': f'{name} 录入成功！({len(embs)}帧)'})


# ========== API: 识别 ==========
@app.route('/api/recognize', methods=['POST'])
def recognize():
    if not gallery_db:
        return jsonify({'success': False, 'message': '人脸库为空，请先录入'})

    embs = []
    for i in range(10):
        key = f'image{i}'
        if key not in request.files:
            break
        img_bytes = request.files[key].read()
        if not img_bytes:
            continue
        try:
            emb = extract_embedding(img_bytes)
            embs.append(emb)
        except Exception:
            continue

    if not embs:
        return jsonify({'success': False, 'message': '未接收到有效图片'})

    probe_emb = torch.stack(embs).mean(dim=0)

    results = []
    for item in gallery_db:
        person_embs = torch.stack(item['embeddings'])
        person_embs = F.normalize(person_embs, p=2, dim=1)
        best = torch.mm(probe_emb, person_embs.t()).max().item()
        results.append({'name': item['name'], 'image': item['image'], 'similarity': round(best, 4)})

    results.sort(key=lambda x: x['similarity'], reverse=True)
    results = results[:TOP_K]

    top = results[0] if results else None
    identified = top and top['similarity'] >= THRESHOLD

    return jsonify({
        'success': True,
        'identified': identified,
        'results': results,
        'threshold': THRESHOLD,
    })


# ========== API: 列表 ==========
@app.route('/api/gallery', methods=['GET'])
def list_gallery():
    return jsonify({
        'success': True,
        'items': [{'name': item['name'], 'image': item['image']} for item in gallery_db],
    })


# ========== API: 删除 ==========
@app.route('/api/gallery/<name>', methods=['DELETE'])
def delete_gallery(name):
    global gallery_db
    gallery_db = [item for item in gallery_db if item['name'] != name]
    save_gallery_db()
    person_dir = GALLERY_DIR / name
    if person_dir.exists():
        import shutil
        shutil.rmtree(person_dir)
    return jsonify({'success': True})


# ========== 静态文件 ==========
@app.route('/gallery/<path:filename>')
def gallery_image(filename):
    return send_from_directory(str(GALLERY_DIR), filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
