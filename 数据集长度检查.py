import os
import numpy as np

def check_fasta(path):
    lengths = []
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                lengths.append(len(line.strip()))
    return lengths

def check_feature_file(path):
    if not os.path.exists(path):
        return None
    try:
        # 读取第一行看看列数
        with open(path, 'r') as f:
            first_line = f.readline().strip().split()
            return len(first_line)
    except:
        return -1

# 路径配置
FASTA_DIR = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/data/E.coli"
FEAT_DIR = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/feature/E.coli_general/train" # 确认你的特征存放路径

print("=== 1. FASTA 序列长度统计 ===")
for name in ["Ecoli_prom.fa", "Ecoli_non_prom.fa"]:
    lens = check_fasta(os.path.join(FASTA_DIR, name))
    if lens:
        print(f"{name}: 样本数={len(lens)}, 平均长度={np.mean(lens):.1f}, 长度标准差={np.std(lens):.1f}")
        if np.std(lens) > 0:
            print(f"  [警告] {name} 长度不统一！这会导致特征提取对齐失败。")
    else:
        print(f"{name}: 文件未找到")

print("\n=== 2. 特征文件维度分析 ===")
# 检查你日志中提到的几个关键文件
feat_files = ["word2vec.txt", "dnabert_features.txt", "cksnap.txt"]
for feat in feat_files:
    # 注意：这里需要替换为你实际提取出的特征文件完整路径
    dim = check_feature_file(os.path.join(FEAT_DIR, feat))
    print(f"特征 {feat}: 维度 = {dim}")
    if feat == "word2vec.txt" and dim and dim > 1000:
        print("  [异常] Word2Vec 维度过高，请检查提取逻辑是否有误。")