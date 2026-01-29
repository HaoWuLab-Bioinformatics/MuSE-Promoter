import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BertConfig


# ==========================================
# 1. 基础工具函数
# ==========================================

def load_fasta_sequences(filename: str):
    """读取 fasta，返回序列列表（忽略 >header 行）"""
    seqs = []
    if not os.path.exists(filename):
        print(f"[Warning] 文件不存在: {filename}")
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


def save_matrix_txt(path: str, mat: torch.Tensor):
    """保存为 txt（空格分隔）"""
    # 自动创建父目录
    os.makedirs(os.path.dirname(path), exist_ok=True)

    mat = mat.detach().cpu()
    with open(path, "w") as f:
        for row in mat:
            # format: 6位有效数字
            f.write(" ".join(f"{x:.6g}" for x in row.tolist()))
            f.write("\n")


# ==========================================
# 2. DNABERT 特有处理逻辑
# ==========================================

def seq2kmer(seq, k=6):
    """将原始序列转换为 DNABERT 需要的 k-mer 字符串格式"""
    kmer_seq = []
    for i in range(len(seq) - k + 1):
        kmer_seq.append(seq[i:i + k])
    return " ".join(kmer_seq)


def extract_features(
        fasta_file: str,
        tokenizer,
        model,
        k: int = 6,
        batch_size: int = 8,
        max_len: int = 512,
        device: str = "cuda"
):
    """
    核心特征提取函数
    """
    # 读取序列
    raw_seqs = load_fasta_sequences(fasta_file)
    if not raw_seqs:
        print("未读取到任何序列，请检查文件内容。")
        return None

    print(f"正在处理: {fasta_file} (共 {len(raw_seqs)} 条序列)...")

    all_features = []

    # 批次处理 (Batch Processing)
    with torch.no_grad():
        for i in range(0, len(raw_seqs), batch_size):
            batch_raw = raw_seqs[i: i + batch_size]

            # 1. 转换为 k-mer 字符串
            batch_kmers = [seq2kmer(s, k=k) for s in batch_raw]

            # 2. Tokenize
            inputs = tokenizer(
                batch_kmers,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len
            )

            # 移动到 GPU
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # 3. 模型前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # 4. 获取特征 (Mean Pooling)
            last_hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            all_features.append(mean_embeddings.cpu())

            # 打印进度
            if (i // batch_size) % 10 == 0 and (i // batch_size) > 0:
                print(f"  -> Batch {i // batch_size} / {len(raw_seqs) // batch_size}")

    # 拼接所有批次
    if not all_features:
        return None

    final_tensor = torch.cat(all_features, dim=0)
    return final_tensor


# ==========================================
# 3. 主程序
# ==========================================

if __name__ == "__main__":
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ---------------- 配置区域 ----------------
    cell_lines = "IMR90"  # 修改你的细胞系名称

    # 本地模型绝对路径
    model_name = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/dnabert_model"

    k_mer = 6
    batch_size = 16
    # ------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"初始化模型: {model_name} on {device}...")

    # 1. 加载模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except Exception as e:
        print("模型加载失败，请检查路径或文件完整性。")
        raise e

    # 2. 设定单一文件路径
    # 对应路径: /home/.../data/R.capsulatus/data.fasta
    input_path = os.path.join(script_dir, "feature", cell_lines, "train/data.fasta")

    # 对应输出: /home/.../feature/R.capsulatus/dnabert_features.txt
    output_path = os.path.join(script_dir, "feature", cell_lines, "train/dnabert_features.txt")

    print(f"\n====== 开始处理单文件: {cell_lines} ======")

    if os.path.exists(input_path):
        features = extract_features(
            input_path,
            tokenizer,
            model,
            k=k_mer,
            batch_size=batch_size,
            device=device
        )

        if features is not None:
            print(f"特征维度: {tuple(features.shape)}")
            save_matrix_txt(output_path, features)
            print(f"已保存至: {output_path}")
    else:
        print(f"错误: 找不到输入文件 -> {input_path}")

    print("\n任务完成。")