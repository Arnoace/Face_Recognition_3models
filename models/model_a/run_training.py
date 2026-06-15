"""简化训练测试 - 28人全量，50x50 图像，150 PCA 成分"""
import cv2, numpy as np, os, sys, time
from pathlib import Path

print("=" * 60, flush=True)
print("Full Training Test (28 persons, 50x50, 150 PCA components)", flush=True)
print("=" * 60, flush=True)

train_dir = "data/cropped_train"
test_dir = "data/cropped_test"

# Step 1: Load data
print("\n[1] Loading training data...", flush=True)
images, labels, label_map = [], [], {}
data_path = Path(train_dir)
person_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
for label_idx, person_dir in enumerate(person_dirs):
    person_name = person_dir.name
    label_map[label_idx] = person_name
    for ext in ['*.jpg', '*.png']:
        for img_path in person_dir.glob(ext):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                labels.append(label_idx)

print(f"   Loaded: {len(images)} images, {len(label_map)} persons", flush=True)

# Step 2: Preprocess
print("\n[2] Preprocessing images (50x50, histogram equalization)...", flush=True)
t0 = time.time()
processed = []
for img in images:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = cv2.resize(gray, (50, 50), interpolation=cv2.INTER_LINEAR)
    gray = cv2.equalizeHist(gray)
    processed.append(gray)
t1 = time.time()
print(f"   Preprocessing done in {t1-t0:.1f}s", flush=True)

# Step 3: Train
print("\n[3] Training Fisherfaces model (num_components=150)...", flush=True)
t0 = time.time()
model = cv2.face.FisherFaceRecognizer_create(num_components=150)
model.train(processed, np.array(labels, dtype=np.int32))
t1 = time.time()
print(f"   Training done in {t1-t0:.1f}s", flush=True)

# Step 4: Save model
print("\n[4] Saving model...", flush=True)
os.makedirs("models", exist_ok=True)
model.write("models/fisherfaces_model.yml")
print("   Model saved.", flush=True)

# Step 5: Test
print("\n[5] Testing on test set...", flush=True)
t0 = time.time()
test_path = Path(test_dir)
reverse_map = {v: k for k, v in label_map.items()}
correct = 0
total = 0
true_names, pred_names = [], []

for person_dir in sorted(test_path.iterdir()):
    if not person_dir.is_dir():
        continue
    true_name = person_dir.name
    label_idx = reverse_map.get(true_name)
    if label_idx is None:
        continue
    for ext in ['*.jpg', '*.png']:
        for img_path in person_dir.glob(ext):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            gray = cv2.resize(gray, (50, 50), interpolation=cv2.INTER_LINEAR)
            gray = cv2.equalizeHist(gray)
            pred_label, conf = model.predict(gray)
            pred_name = label_map.get(pred_label, f"Unknown-{pred_label}")
            true_names.append(true_name)
            pred_names.append(pred_name)
            total += 1
            if pred_name == true_name:
                correct += 1

t1 = time.time()
accuracy = correct / total * 100 if total > 0 else 0
print(f"   Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%", flush=True)
print(f"   Testing time: {t1-t0:.1f}s ({total/(t1-t0):.1f} images/s)", flush=True)

# Step 6: Quick metrics
from sklearn.metrics import classification_report, confusion_matrix
print("\n[6] Classification Report:", flush=True)
print(classification_report(true_names, pred_names, zero_division=0), flush=True)

print("\nDone!", flush=True)
