import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', message='.*estimate is deprecated.*')
import os, sys, json, time, random, traceback
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.arcface import ArcFaceModel
from backend import config

TRAIN_DIR = os.path.join(config.BASE_DIR, 'processed', 'cropped_train')
TEST_DIR = os.path.join(config.BASE_DIR, 'processed', 'cropped_test')
FEAT_OUT = config.FEATURES_DIR
os.makedirs(FEAT_OUT, exist_ok=True)
RESULT_DIR = os.path.join(config.DATA_DIR, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

MAX_PER_PERSON = 100  # 训练每人取100张 (原340)
TEST_PER_PERSON = 100  # 测试每人取100张 (某些人不足100则全部取)
RANDOM_SEED = 42       # 固定随机种子, 结果可复现


def features_exist():
    """检查是否已有训练好的特征文件."""
    return (os.path.exists(os.path.join(FEAT_OUT, 'arcface_features.npy')) and
            os.path.exists(os.path.join(FEAT_OUT, 'arcface_labels.npy')) and
            os.path.exists(os.path.join(FEAT_OUT, 'arcface_label_names.json')))


def load_features():
    """加载已保存的特征文件."""
    features = np.load(os.path.join(FEAT_OUT, 'arcface_features.npy'))
    labels = np.load(os.path.join(FEAT_OUT, 'arcface_labels.npy'))
    with open(os.path.join(FEAT_OUT, 'arcface_label_names.json'), 'r') as f:
        label_names = json.load(f)
    return features, labels, label_names


# -------------------------------------------------------
# 第1步: 特征提取 (带进度条)
# -------------------------------------------------------
def extract_features(model, data_dir, max_per_person=MAX_PER_PERSON):
    identities = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"    发现 {len(identities)} 个身份目录")
    features, label_names = [], {}
    total_ok, start_time = 0, time.time()
    for idx, identity in enumerate(identities):
        identity_dir = os.path.join(data_dir, identity)
        image_files = [f for f in os.listdir(identity_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.seed(RANDOM_SEED)
        if max_per_person and len(image_files) > max_per_person:
            image_files = random.sample(image_files, max_per_person)
        person_feats = []
        for img_file in image_files:
            img = cv2.imread(os.path.join(identity_dir, img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            try:
                img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                feat = model.extract_feature(img_3ch)
                person_feats.append(feat)
                total_ok += 1
            except Exception as e:
                # ONNX Runtime 或图片格式问题, 跳过该图
                print(f"      [跳过] {img_file}: {type(e).__name__}: {str(e)[:60]}")
                continue
        if person_feats:
            mean_feat = np.mean(person_feats, axis=0)
            mean_feat = mean_feat / np.linalg.norm(mean_feat)
            features.append(mean_feat)
            label_names[str(idx)] = identity
        elapsed = time.time() - start_time
        rate = total_ok / elapsed if elapsed > 0 else 0
        eta = (len(identities)-idx-1) * (elapsed/(idx+1))/60 if idx > 0 else 0
        bar_len=30; filled=int(bar_len*(idx+1)/len(identities))
        print(f"  [{'#'*filled}{'-'*(bar_len-filled)}] {idx+1}/{len(identities)} | {identity}: {len(person_feats)}张 | 共{total_ok}张 | {rate:.1f}张/秒 | ETA {eta:.1f}分")
    print(f"    特征提取完成! 耗时{time.time()-start_time:.1f}秒")
    return np.array(features), np.array(list(range(len(features)))), label_names


# -------------------------------------------------------
# 第2步: 评估 (带进度条) + 可视化
# -------------------------------------------------------
def evaluate_and_visualize(model, features, labels, label_names, test_dir, max_per_person=TEST_PER_PERSON):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif']=['SimHei','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
        HAS_MPL=True
    except ImportError:
        print("    [提示] matplotlib未安装,跳过图表 (pip install matplotlib)"); HAS_MPL=False

    identities = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    total_test_imgs = 0
    for iden in identities:
        files = [f for f in os.listdir(os.path.join(test_dir, iden)) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        total_test_imgs += min(max_per_person, len(files)) if max_per_person else len(files)

    ic = {i:0 for i in identities}; it = {i:0 for i in identities}
    conf = {i:{j:0 for j in range(len(identities))} for i in range(len(identities))}
    total=correct=top3=0; samples=[]; skipped=0
    print(f"    测试集共约 {total_test_imgs} 张图片, {len(identities)} 个身份")
    print("    开始评估..."); t0=time.time()

    for idx, identity in enumerate(identities):
        files = [f for f in os.listdir(os.path.join(test_dir,identity)) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        random.seed(RANDOM_SEED)
        if max_per_person: files = random.sample(files, min(max_per_person, len(files)))

        for f in files:
            img = cv2.imread(os.path.join(test_dir,identity,f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped += 1
                continue
            try:
                feat = model.extract_feature(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            except Exception as e:
                skipped += 1
                continue
            total+=1; it[identity]+=1; feat/=np.linalg.norm(feat)
            sims=np.dot(features,feat); best=int(np.argmax(sims)); si=np.argsort(sims)[::-1]
            pred=label_names.get(str(int(labels[best])),'?'); conf[idx][best]+=1
            if pred==identity: correct+=1; ic[identity]+=1
            if identity in [label_names.get(str(int(labels[i])),'') for i in si[:3]]: top3+=1
            if len(samples)<60: samples.append({'p':os.path.join(test_dir,identity,f),'t':identity,'p_':pred,'c':pred==identity,'s':float(sims[best])})

        # 进度条 (每个人完成后更新)
        bar_len=30; filled=int(bar_len*(idx+1)/len(identities))
        elapsed=time.time()-t0; rate=total/elapsed if elapsed>0 else 0
        eta=(len(identities)-idx-1)*(elapsed/(idx+1))/60 if idx>0 else 0
        print(f"  [{'#'*filled}{'-'*(bar_len-filled)}] {idx+1}/{len(identities)} | {identity} | 已评估{total}张 | {rate:.1f}张/秒 | ETA {eta:.1f}分")

    print(f"    评估完成! 成功{total}张, 跳过{skipped}张, 耗时{time.time()-t0:.1f}秒")
    if total>0:
        print(f"\n  === 评估结果 ===")
        print(f"  测试集: {test_dir}")
        print(f"  Top-1 准确率: {correct/total*100:.2f}% ({correct}/{total})")
        print(f"  Top-3 准确率: {top3/total*100:.2f}% ({top3}/{total})")
        print(f"  身份数: {len(features)} | 特征维度: {features.shape[1]} | 相似度阈值: {config.SIMILARITY_THRESHOLD}")

    if not HAS_MPL: return
    os.makedirs(RESULT_DIR, exist_ok=True)
    print("\n  生成可视化图表...")

    # 图1: 每人准确率柱状图
    print("    [1/4] 每人准确率柱状图...")
    accs=[ic[i]/it[i]*100 if it[i]>0 else 0 for i in identities]
    fig,ax=plt.subplots(figsize=(14,6))
    bars=ax.bar(range(len(identities)),accs,color=['#22c55e' if a>=50 else '#ef4444' for a in accs],edgecolor='white')
    ax.set_xticks(range(len(identities))); ax.set_xticklabels(list(identities),rotation=45,ha='right',fontsize=8)
    ax.set_ylabel('准确率(%)'); ax.set_title('每人识别准确率 (ArcFace on Yale B)'); ax.set_ylim(0,105)
    ax.axhline(y=correct/total*100,color='#3b82f6',ls='--',label=f'总体: {correct/total*100:.1f}%'); ax.legend()
    for b,a in zip(bars,accs): ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{a:.0f}%',ha='center',va='bottom',fontsize=7)
    plt.tight_layout(); fig.savefig(os.path.join(RESULT_DIR,'accuracy_per_identity.png'),dpi=150); plt.close(fig)
    print(f"      -> {RESULT_DIR}\\accuracy_per_identity.png")

    # 图2: 混淆矩阵
    print("    [2/4] 混淆矩阵...")
    cm=np.zeros((len(identities),len(identities)),dtype=int)
    for i in range(len(identities)):
        for j in range(len(identities)): cm[i][j]=conf[i].get(j,0)
    cnm=np.zeros_like(cm,dtype=float)
    for i in range(len(identities)):
        s=cm[i].sum(); cnm[i]=cm[i]/s*100 if s>0 else 0
    fig,ax=plt.subplots(figsize=(18,16))
    im=ax.imshow(cnm,cmap='Blues',vmin=0,vmax=100)
    ax.set_xticks(range(len(identities))); ax.set_yticks(range(len(identities)))
    ax.set_xticklabels(list(identities),rotation=90,fontsize=5); ax.set_yticklabels(list(identities),fontsize=5)
    ax.set_xlabel('预测标签'); ax.set_ylabel('真实标签'); ax.set_title('混淆矩阵 (ArcFace on Yale B)')
    plt.colorbar(im,ax=ax,shrink=0.7,label='%'); plt.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR,'confusion_matrix.png'),dpi=150); plt.close(fig)
    print(f"      -> {RESULT_DIR}\\confusion_matrix.png")

    # 图3: 相似度分布
    print("    [3/4] 相似度分布图...")
    sims_data=[]
    for idx,identity in enumerate(identities):
        files=[f for f in os.listdir(os.path.join(test_dir,identity)) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        random.seed(RANDOM_SEED); files=random.sample(files,min(20,len(files)))
        for f in files:
            img=cv2.imread(os.path.join(test_dir,identity,f),cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            try: feat=model.extract_feature(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR))
            except Exception: continue
            feat/=np.linalg.norm(feat); s=np.dot(features,feat)
            sims_data.append({'c':float(s[identities.index(identity)]),'b':float(np.max(s))})
    fig,ax=plt.subplots(figsize=(10,5))
    cs=[s['c'] for s in sims_data]; bs=[s['b'] for s in sims_data]
    ax.hist(cs,bins=30,alpha=0.7,color='#22c55e',label=f'正确匹配 (均值{np.mean(cs):.3f})')
    ax.hist(bs,bins=30,alpha=0.3,color='#3b82f6',label=f'最佳匹配 (均值{np.mean(bs):.3f})')
    ax.axvline(x=config.SIMILARITY_THRESHOLD,color='red',ls='--',label=f'阈值={config.SIMILARITY_THRESHOLD}')
    ax.set_xlabel('余弦相似度'); ax.set_ylabel('频次'); ax.set_title('相似度分布'); ax.legend()
    plt.tight_layout(); fig.savefig(os.path.join(RESULT_DIR,'similarity_distribution.png'),dpi=150); plt.close(fig)
    print(f"      -> {RESULT_DIR}\\similarity_distribution.png")

    # 图4: 样本识别结果网格
    print("    [4/4] 样本识别结果网格...")
    ss=random.Random(RANDOM_SEED).sample(samples,min(40,len(samples)))
    gs=int(np.ceil(np.sqrt(len(ss))))
    fig,axes=plt.subplots(gs,gs,figsize=(gs*2.5,gs*2.5)); axes=axes.flatten()
    for i in range(gs*gs):
        axes[i].axis('off')
        if i<len(ss):
            img=cv2.imread(ss[i]['p'],cv2.IMREAD_GRAYSCALE)
            if img is not None: axes[i].imshow(img,cmap='gray')
            axes[i].set_title(f"T:{ss[i]['t']}\nP:{ss[i]['p_']}\n{ss[i]['s']:.2f}",fontsize=6,color='green' if ss[i]['c'] else 'red')
    fig.suptitle('测试集人脸识别结果',fontsize=14,y=1.01)
    plt.tight_layout(); fig.savefig(os.path.join(RESULT_DIR,'recognition_samples.png'),dpi=150); plt.close(fig)
    print(f"      -> {RESULT_DIR}\\recognition_samples.png")
    print(f"\n    所有图表已保存至: {RESULT_DIR}")


# -------------------------------------------------------
# 主流程
# -------------------------------------------------------
def main():
    print('\n'+'='*50)
    print('  ArcFace 人脸特征提取 + 评估 + 可视化')
    print('='*50)

    # 检查是否已有特征, 询问用户
    need_train = True
    if features_exist():
        print('\n  检测到已保存的特征文件:')
        feats = np.load(os.path.join(FEAT_OUT, 'arcface_features.npy'))
        with open(os.path.join(FEAT_OUT, 'arcface_label_names.json'), 'r') as f:
            names = json.load(f)
        print(f'    arcface_features.npy   ({feats.shape[0]} 个身份)')
        print(f'    arcface_label_names.json ({len(names)} 个名称)')
        print()
        print('  请选择:')
        print('    [1] 重新训练 + 评估 (从头跑)')
        print('    [2] 跳过训练, 直接评估 (加载已有特征)')
        choice = input('  输入 1 或 2 (默认 2): ').strip()
        need_train = (choice == '1')
        print(f'  选择: {"重新训练" if need_train else "直接评估"}')
    else:
        print('\n  未发现特征文件, 将执行完整训练+评估流程')

    # 加载 ArcFace 模型
    print('\n[1/4] 加载 ArcFace 模型...')
    print('  (首次运行会自动下载 ~100MB 模型文件, 请等待)')
    sys.stdout.flush()
    model = ArcFaceModel()
    print(f'  模型: {model.name} | 特征维度: {model.feature_dim} | OK!')

    # 训练 或 加载已有特征
    if need_train:
        print(f'\n[2/4] 提取训练集特征...')
        print(f'  目录: {TRAIN_DIR}')
        print(f'  每人 {MAX_PER_PERSON} 张 | 共 28 人')
        sys.stdout.flush()
        features, labels, label_names = extract_features(model, TRAIN_DIR, MAX_PER_PERSON)
        if len(features)==0:
            print('[错误] 未提取到特征! 检查路径:', TRAIN_DIR)
            return
        print(f'\n[3/4] 保存特征到 {FEAT_OUT}')
        np.save(os.path.join(FEAT_OUT,'arcface_features.npy'), features)
        np.save(os.path.join(FEAT_OUT,'arcface_labels.npy'), labels)
        with open(os.path.join(FEAT_OUT,'arcface_label_names.json'),'w') as f: json.dump(label_names,f)
        print(f'  arcface_features.npy    ({features.shape})')
        print(f'  arcface_labels.npy      ({labels.shape})')
        print(f'  arcface_label_names.json ({len(label_names)} 个身份)')
    else:
        print(f'\n[2/4] 加载已有特征文件...')
        features, labels, label_names = load_features()
        print(f'  已加载: {features.shape[0]} 个身份, 维度 {features.shape[1]}')
        print(f'  [3/4] 跳过 (保留特征文件)')

    # 评估 + 可视化
    print(f'\n[4/4] 评估 + 可视化...')
    print(f'  测试集: {TEST_DIR}')
    print(f'  (每人取 {TEST_PER_PERSON} 张, 不足则全取)')
    sys.stdout.flush()
    evaluate_and_visualize(model, features, labels, label_names, TEST_DIR, TEST_PER_PERSON)

    print(f'\n{"="*50}')
    print(f'  全部完成!')
    print(f'  特征文件: {FEAT_OUT}')
    print(f'  结果图表: {RESULT_DIR}')
    print(f'  启动后端: & ".\\venv\\Scripts\\python.exe" run.py')
    print(f'{"="*50}\n')


if __name__ == '__main__':
    main()
