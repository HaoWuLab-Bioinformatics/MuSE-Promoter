import os
import torch
import torch.nn as nn


def load_fasta_sequences(filename: str):
    """读取 fasta，返回序列列表（忽略 >header 行）"""
    seqs = []
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
    with open(index_file, "r") as f:
        tokens = f.read().strip().split()
    return tokens


def load_word2vec_txt(path: str, dtype=torch.float32):
    """
    纯 torch 读取 word2vec_promoters.txt（空白分隔的矩阵）
    返回: Tensor [V, D]
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
    return torch.tensor(rows, dtype=dtype)


def save_matrix_txt(path: str, mat: torch.Tensor):
    """用标准库保存为 txt（空格分隔），兼容你原 np.savetxt 输出格式"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mat = mat.detach().cpu()
    with open(path, "w") as f:
        for row in mat:
            # row: (D,)
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
    输入:
      fasta_file: fasta 文件
      index_list: kmer 字典 list（你的 index_promoters.txt）
      word2vec: Tensor [V, D]
    输出:
      features: Tensor [N, out_len]
    """
    # 建 kmer->id 的 dict（比 index.index 快很多）
    kmer_to_id = {kmer: i for i, kmer in enumerate(index_list)}

    seqs = load_fasta_sequences(fasta_file)
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
                # 如果出现字典外 kmer：可以跳过或用 0 代替
                # 这里用 0 代替，避免长度变 0
                ids.append(0)

        if len(ids) == 0:
            # 极端情况：序列比 k 还短
            # 直接用全 0
            feats.append(torch.zeros(out_len, dtype=torch.float32))
            continue

        ids_t = torch.tensor(ids, dtype=torch.long)             # (L,)
        emb = word2vec[ids_t]                                   # (L, D)

        # 2) 按你原逻辑：把每个 kmer 的 embedding 展平为 1D
        flat = emb.reshape(-1)                                   # (L*D,)

        # 3) AdaptiveAvgPool1d 需要 (N,C,L)，你原来是 (1,1,L)
        x = flat.view(1, 1, -1)                                  # (1,1,L*D)
        x = pool(x)                                              # (1,1,out_len)
        out = x.view(-1)                                         # (out_len,)
        feats.append(out)

    return torch.stack(feats, dim=0)  # (N, out_len)


if __name__ == "__main__":
    cell_lines = "mouse nonTATA"
    fasta_file = f"data/{cell_lines}/test/data.fasta"
    index_file = "index_promoters.txt"
    w2v_file = "word2vec_promoters.txt"
    out_file = f"feature/{cell_lines}/word2vec.txt"

    index_list = read_index_list(index_file)
    word2vec = load_word2vec_txt(w2v_file)  # Tensor [V,D]

    features = word_embedding(fasta_file, index_list, word2vec, k=5, out_len=1000 * 8)
    print("feature shape:", tuple(features.shape))

    save_matrix_txt(out_file, features)
    print("saved to:", out_file)
