import os
import torch
import torch.nn as nn
import sys

# ================= 配置区域 =================
# 1. 基础路径配置
BASE_PROJECT_DIR = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main"
CELL_NAME = "HUVEC"

# 2. word2vec 预训练文件路径 (请确保这两个文件存在于项目根目录或指定位置)
# 假设它们位于 BASE_PROJECT_DIR 下，或者是当前目录
INDEX_FILE = os.path.join(BASE_PROJECT_DIR, "index_promoters.txt")
W2V_FILE = os.path.join(BASE_PROJECT_DIR, "word2vec_promoters.txt")

# 3. 输出特征的根目录
FEATURE_ROOT = os.path.join(BASE_PROJECT_DIR, "EPfeature", CELL_NAME)


# ===========================================

def load_fasta_sequences(filename: str):
    """读取 fasta，返回序列列表（忽略 >header 行）"""
    seqs = []
    if not os.path.exists(filename):
        print(f"[WARN] 文件不存在: {filename}")
        return []

    with open(filename, "r") as f:
        cur = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line)
        if cur:
            seqs.append("".join(cur))
    return seqs


def read_index_list(index_file: str):
    """读取 index_promoters.txt（空格分隔）"""
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"找不到 Index 文件: {index_file}")
    with open(index_file, "r") as f:
        tokens = f.read().strip().split()
    return tokens


def load_word2vec_txt(path: str, dtype=torch.float32):
    """
    纯 torch 读取 word2vec_promoters.txt（空白分隔的矩阵）
    返回: Tensor [V, D]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 Word2Vec 文件: {path}")

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    return torch.tensor(rows, dtype=dtype)


def save_matrix_txt(path: str, mat: torch.Tensor):
    """保存为 txt（空格分隔）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mat = mat.detach().cpu()
    with open(path, "w") as f:
        for row in mat:
            f.write(" ".join(f"{x:.6g}" for x in row.tolist()))
            f.write("\n")


def word_embedding(
        fasta_file: str,
        index_list,
        word2vec: torch.Tensor,
        k: int = 5,
        out_len: int = 1000 * 8,
):
    """
    生成 Word2Vec 特征
    """
    # 建 kmer->id 的 dict（比 index.index 快很多）
    kmer_to_id = {kmer: i for i, kmer in enumerate(index_list)}

    seqs = load_fasta_sequences(fasta_file)
    if not seqs:
        return None

    pool = nn.AdaptiveAvgPool1d(out_len)

    feats = []
    for seq in seqs:
        # 1) 生成 kmer id 序列
        ids = []
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if kmer in kmer_to_id:
                ids.append(kmer_to_id[kmer])
            else:
                # 如果出现字典外 kmer，用 0 代替（或者可以处理为 UNK）
                ids.append(0)

        if len(ids) == 0:
            # 极端情况：序列比 k 还短，填充 0
            feats.append(torch.zeros(out_len, dtype=torch.float32))
            continue

        ids_t = torch.tensor(ids, dtype=torch.long)  # (L,)
        emb = word2vec[ids_t]  # (L, D)

        # 2) 展平
        flat = emb.reshape(-1)  # (L*D,)

        # 3) AdaptiveAvgPool1d
        x = flat.view(1, 1, -1)  # (1,1,L*D)
        x = pool(x)  # (1,1,out_len)
        out = x.view(-1)  # (out_len,)
        feats.append(out)

    if not feats:
        return None

    return torch.stack(feats, dim=0)  # (N, out_len)


def main():
    print(f"--- 开始处理 Cell: {CELL_NAME} ---")
    print(f"Index 文件: {INDEX_FILE}")
    print(f"W2V 文件: {W2V_FILE}")
    print(f"Feature 根目录: {FEATURE_ROOT}")

    # 1. 加载资源 (只加载一次，节省时间)
    print("正在加载 Word2Vec 模型和 Index...")
    try:
        index_list = read_index_list(INDEX_FILE)
        word2vec_mat = load_word2vec_txt(W2V_FILE)  # Tensor [V,D]
        print(f"资源加载完毕。词表大小: {len(index_list)}, 向量维度: {word2vec_mat.shape}")
    except Exception as e:
        print(f"[ERROR] 加载资源失败: {e}")
        print("请检查 index_promoters.txt 和 word2vec_promoters.txt 是否在正确路径。")
        return

    # 2. 遍历 enhancer 和 promoter
    sub_types = ["enhancers", "promoters"]
    splits = ["train", "test"]

    for sub_type in sub_types:
        for split in splits:
            # 根据上一轮代码，FASTA 文件通常位于 split_data 文件夹中
            # 路径构建: EPfeature/GM12878/<sub_type>/split_data/<split>/data.fasta
            fasta_path = os.path.join(FEATURE_ROOT, sub_type, "split_data", split, "data.fasta")

            # 输出文件路径: EPfeature/GM12878/<sub_type>/<split>/word2vec.txt
            out_file = os.path.join(FEATURE_ROOT, sub_type, split, "word2vec.txt")

            # 检查输入是否存在
            if not os.path.exists(fasta_path):
                # 备用方案：如果 split_data 目录结构不同，尝试直接在 feature 目录下找
                alt_path = os.path.join(FEATURE_ROOT, sub_type, split, "data.fasta")
                if os.path.exists(alt_path):
                    fasta_path = alt_path
                else:
                    print(f"[SKIP] 找不到输入文件: {fasta_path}")
                    continue

            print(f"\n正在处理: {sub_type} - {split}")
            print(f"  输入: {fasta_path}")

            # 生成特征
            features = word_embedding(
                fasta_path,
                index_list,
                word2vec_mat,
                k=5,
                out_len=1000 * 8
            )

            if features is not None:
                print(f"  特征形状: {tuple(features.shape)}")
                save_matrix_txt(out_file, features)
                print(f"  [OK] 保存至: {out_file}")
            else:
                print("  [WARN] 生成失败或序列为空")

    print("\n--- 全部完成 ---")


if __name__ == "__main__":
    main()