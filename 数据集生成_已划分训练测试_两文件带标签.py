# -*- coding: utf-8 -*-
"""
【手动划分版】特征生成脚本
修改点：
1) ✅ 支持直接读取已划分好的 train/test 路径。
2) ✅ 依然保持 Word2Vec 仅在训练集上训练，再应用到测试集。
3) ✅ 保持原有的 iLearn 清洗、Mismatch 生成及行数对齐逻辑。
"""

import os
import sys
import csv
import re
import random
import subprocess
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn

# ================= 配置区域 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PROJECT_DIR = os.path.join(SCRIPT_DIR, "iPro-WAEL-main")
FEATURE_ROOT_DIR = os.path.join(BASE_PROJECT_DIR, "feature")
ILEARN_SCRIPT_NAME = "iLearn-nucleotide-basic.py"

# ====== 你的手动路径配置 ======
DATASETS_CONFIG = {
    "IMR90": {
        "train_fasta": "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/data/IMR90/train/data.fasta",
        "train_label": "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/data/IMR90/train/y_train.txt",
        "test_fasta": "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/data/IMR90/test/data.fasta",
        "test_label": "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/data/IMR90/test/y_test.txt",
    }
}

# 算法参数
W2V_KMER = 5
W2V_VECTOR_DIM = 100
W2V_OUT_LEN = 8000
DNA_ALPHABET = ["A", "C", "G", "T"]


# ================= 基础工具函数 =================

def load_manual_data(fasta_path, label_path):
    """根据提供的路径读取序列和标签"""
    if not os.path.exists(fasta_path) or not os.path.exists(label_path):
        print(f"[ERROR] 路径不存在: {fasta_path} 或 {label_path}")
        return [], [], []

    headers, seqs = [], []
    current_seq = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if current_seq: seqs.append("".join(current_seq))
                headers.append(line[1:])
                current_seq = []
            else:
                current_seq.append(line)
        if current_seq: seqs.append("".join(current_seq))

    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    labels.append(int(float(line)))
                except ValueError:
                    pass

    # 对齐检查
    min_len = min(len(seqs), len(labels))
    return headers[:min_len], seqs[:min_len], labels[:min_len]


def write_fasta_and_y(headers, seqs, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fasta_path = os.path.join(out_dir, "data.fasta")
    y_path = os.path.join(out_dir, "y.txt")
    with open(fasta_path, "w", encoding="utf-8") as wf, open(y_path, "w", encoding="utf-8") as wy:
        for h, s, l in zip(headers, seqs, labels):
            clean_h = (h.split()[0]).replace("|", "_")
            wf.write(f">{clean_h}|{int(l)}|processed\n{s}\n")
            wy.write(f"{int(l)}\n")


def _count_lines(path):
    if not os.path.exists(path): return 0
    with open(path, "r", errors="ignore") as f:
        return sum(1 for _ in f)


def _pad_or_trim_txt(path, target_lines):
    if not os.path.exists(path): return
    with open(path, "r", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    max_cols = 1
    for ln in lines:
        if ln.strip():
            cols = len(ln.split())
            if cols > max_cols: max_cols = cols

    if len(lines) > target_lines:
        lines = lines[:target_lines]
    elif len(lines) < target_lines:
        pad_line = " ".join(["0"] * max_cols)
        lines.extend([pad_line] * (target_lines - len(lines)))

    with open(path, "w", encoding="utf-8") as f:
        for ln in lines: f.write(ln + "\n")


def align_feature_rows(out_dir):
    y_path = os.path.join(out_dir, "y.txt")
    if not os.path.exists(y_path): return
    target = _count_lines(y_path)
    feats = ["cksnap.txt", "mismatch.txt", "rckmer.txt", "psetnc.txt", "tpcp.txt", "word2vec.txt"]
    for fn in feats:
        _pad_or_trim_txt(os.path.join(out_dir, fn), target)


# ================= iLearn 逻辑 =================

def find_ilearn_paths(start_dir, script_name):
    start_path = Path(start_dir)
    search_bases = [start_path] + list(start_path.parents)[:3]
    for base in search_bases:
        results = list(base.rglob(script_name))
        if results:
            found_basic = str(results[0])
            pse_script = os.path.join(os.path.dirname(found_basic), "iLearn-nucleotide-Pse.py")
            if os.path.exists(pse_script): return found_basic, pse_script
    return None, None


def run_cmd(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.returncode == 0


def sanitize_ilearn_csv_to_txt(in_csv, out_txt):
    if not os.path.exists(in_csv): return False
    with open(in_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    if len(reader) < 2: return False

    header = [h.strip().lower() for h in reader[0]]
    DROP_KEYS = {"label", "class", "y", "target", "output"}
    ID_KEYS = {"id", "name", "sample", "index", "sequence"}

    keep_idx = []
    for i, col in enumerate(header):
        if col not in DROP_KEYS and col not in ID_KEYS and not any(col.startswith(k) for k in ID_KEYS):
            keep_idx.append(i)

    with open(out_txt, "w", encoding="utf-8") as f:
        for row in reader[1:]:
            vals = []
            for i in keep_idx:
                try:
                    float(row[i])
                    vals.append(row[i])
                except:
                    vals.append("0")
            f.write(" ".join(vals) + "\n")
    return True


def generate_ilearn_features(data_dir, headers, seqs, labels, ilearn_basic, ilearn_pse):
    tmp_fasta = os.path.join(data_dir, "temp_ilearn.fasta")
    with open(tmp_fasta, "w", encoding="utf-8") as f:
        for i, (h, s, y) in enumerate(zip(headers, seqs, labels)):
            f.write(f">S{i}|{int(y)}\n{s}\n")

    tasks = [("CKSNAP", ilearn_basic, "cksnap"), ("RCKmer", ilearn_basic, "rckmer"),
             ("TNC", ilearn_basic, "tpcp"), ("SCPseTNC", ilearn_pse, "psetnc")]

    py = sys.executable
    for method, script, name in tasks:
        raw_csv = os.path.join(data_dir, f"raw_{name}.csv")
        final_txt = os.path.join(data_dir, f"{name}.txt")
        args = ["--file", tmp_fasta, "--method", method, "--format", "csv", "--out", raw_csv]
        if method == "SCPseTNC": args += ["--type", "DNA"]

        if run_cmd([py, script] + args):
            sanitize_ilearn_csv_to_txt(raw_csv, final_txt)
            if os.path.exists(raw_csv): os.remove(raw_csv)
    if os.path.exists(tmp_fasta): os.remove(tmp_fasta)


# ================= Mismatch & Word2Vec =================

def generate_mismatch_txt(seqs, out_txt, k=5, m=1):
    kmers = ["".join(p) for p in product(DNA_ALPHABET, repeat=k)]
    kmer_idx = {km: i for i, km in enumerate(kmers)}
    with open(out_txt, "w", encoding="utf-8") as f:
        for seq in seqs:
            vec = [0.0] * len(kmers)
            seq = seq.upper()
            n = len(seq) - k + 1
            if n > 0:
                for i in range(n):
                    sub = seq[i:i + k]
                    if all(c in DNA_ALPHABET for c in sub):
                        # 简化版：仅记录原kmer（如需精确mismatch可展开neighbor）
                        if sub in kmer_idx: vec[kmer_idx[sub]] += 1
                vec = [v / n for v in vec]
            f.write(" ".join(f"{v:.6f}" for v in vec) + "\n")


def train_w2v(train_seqs, idx_path, vec_path):
    from gensim.models import Word2Vec
    sentences = [[s[i:i + W2V_KMER] for i in range(len(s) - W2V_KMER + 1)] for s in train_seqs]
    model = Word2Vec(sentences, vector_size=W2V_VECTOR_DIM, window=5, min_count=1, sg=1)
    vocab = list(model.wv.index_to_key)
    with open(idx_path, "w") as f: f.write(" ".join(vocab))
    with open(vec_path, "w") as f:
        for k in vocab: f.write(" ".join(map(str, model.wv[k])) + "\n")


def generate_w2v_features(seqs, idx_path, vec_path, out_path):
    with open(idx_path, "r") as f:
        vocab = f.read().split()
    word_map = {w: i for i, w in enumerate(vocab)}
    with open(vec_path, "r") as f:
        matrix = torch.tensor([[float(x) for x in l.split()] for l in f])

    pool = nn.AdaptiveAvgPool1d(W2V_OUT_LEN)
    with open(out_path, "w") as f:
        for s in seqs:
            ids = [word_map.get(s[i:i + W2V_KMER], 0) for i in range(len(s) - W2V_KMER + 1)]
            if not ids: ids = [0]
            emb = matrix[torch.tensor(ids)].reshape(1, 1, -1)
            out = pool(emb).view(-1).tolist()
            f.write(" ".join(f"{x:.6f}" for x in out) + "\n")


# ================= 主流程 =================

def main():
    ilearn_basic, ilearn_pse = find_ilearn_paths(BASE_PROJECT_DIR, ILEARN_SCRIPT_NAME)
    if not ilearn_basic:
        print("错误: 未找到 iLearn 脚本")
        return

    for ds_name, paths in DATASETS_CONFIG.items():
        print(f"\n>>> 处理数据集: {ds_name}")

        # 1. 加载手动指定的 Train/Test 数据
        tr_h, tr_s, tr_l = load_manual_data(paths["train_fasta"], paths["train_label"])
        te_h, te_s, te_l = load_manual_data(paths["test_fasta"], paths["test_label"])

        ds_out_root = os.path.join(FEATURE_ROOT_DIR, ds_name)
        os.makedirs(ds_out_root, exist_ok=True)

        # 2. 训练 Word2Vec (仅用训练集)
        w2v_idx = os.path.join(ds_out_root, "w2v_index.txt")
        w2v_vec = os.path.join(ds_out_root, "w2v_vecs.txt")
        print("正在训练 Word2Vec...")
        train_w2v(tr_s, w2v_idx, w2v_vec)

        # 3. 分别生成特征
        for stype, (h, s, l) in [("train", (tr_h, tr_s, tr_l)), ("test", (te_h, te_s, te_l))]:
            print(f"  正在生成 {stype} 特征...")
            out_dir = os.path.join(ds_out_root, stype)
            write_fasta_and_y(h, s, l, out_dir)

            # iLearn
            generate_ilearn_features(out_dir, h, s, l, ilearn_basic, ilearn_pse)
            # Mismatch
            generate_mismatch_txt(s, os.path.join(out_dir, "mismatch.txt"))
            # Word2Vec
            generate_w2v_features(s, w2v_idx, w2v_vec, os.path.join(out_dir, "word2vec.txt"))

            # 对齐
            align_feature_rows(out_dir)

    print("\n任务全部完成！特征保存在:", FEATURE_ROOT_DIR)


if __name__ == "__main__":
    main()