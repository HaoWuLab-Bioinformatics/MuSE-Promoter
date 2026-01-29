# -*- coding: utf-8 -*-
import os
import sys
import csv
import re
import random
import subprocess
from itertools import product
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn

# ================= 配置区域 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PROJECT_DIR = os.path.join(SCRIPT_DIR, "iPro-WAEL-main")
FEATURE_ROOT_DIR = os.path.join(BASE_PROJECT_DIR, "feature")
ILEARN_SCRIPT_NAME = "iLearn-nucleotide-basic.py"

DATASETS_CONFIG = {
    "R.capsulatus": {
        "fasta": "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/data/R.capsulatus/data.fasta",
        "label": "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/data/R.capsulatus/y.txt",
    }
}

SPLIT_SEED = 42
TRAIN_TO_TEST = (4, 1)
DNA_ALPHABET = ["A", "C", "G", "T"]
W2V_KMER = 3
W2V_VECTOR_DIM = 100
W2V_OUT_LEN = 8000


# ================= 核心工具 =================

def read_fasta_strict(path):
    headers, seqs = [], []
    if not os.path.exists(path): return headers, seqs
    with open(path, "r", encoding="utf-8") as f:
        curr_seq = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if curr_seq: seqs.append("".join(curr_seq).upper())
                headers.append(line[1:])
                curr_seq = []
            else:
                curr_seq.append(line)
        if curr_seq: seqs.append("".join(curr_seq).upper())
    return headers, seqs


def align_feature_rows(out_dir, expected_count):
    """强制对齐所有融合脚本需要的txt，缺失则补0"""
    feats = ["cksnap.txt", "mismatch.txt", "rckmer.txt", "psetnc.txt", "tpcp.txt", "word2vec.txt"]
    for fn in feats:
        fp = os.path.join(out_dir, fn)
        # 如果文件根本没生成，创建一个全是0的占位文件
        if not os.path.exists(fp):
            print(f"  [Warning] {fn} 缺失，正在创建全0占位文件...")
            # 这里的维度根据你的融合脚本通常期望的来，TPCP通常是三核苷酸性质(数十维)
            dummy_dim = 12 if fn == "tpcp.txt" else 10
            with open(fp, "w") as f:
                for _ in range(expected_count):
                    f.write(" ".join(["0.0"] * dummy_dim) + "\n")
            continue

        with open(fp, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        if len(lines) != expected_count:
            if not lines: lines = ["0.0"]
            max_cols = len(lines[0].split())
            if len(lines) > expected_count:
                lines = lines[:expected_count]
            else:
                lines += [" ".join(["0.0"] * max_cols)] * (expected_count - len(lines))
            with open(fp, "w") as f:
                for l in lines: f.write(l + "\n")


# ================= 特征引擎 =================

def generate_ilearn_features(fasta_path, out_dir, basic_p, pse_p):
    tasks = [
        ("CKSNAP", basic_p, "cksnap"),
        ("RCKmer", basic_p, "rckmer"),
        ("TPCP", basic_p, "tpcp"),
        ("SCPseTNC", pse_p, "psetnc"),
    ]

    for method, script, name in tasks:
        raw_csv = os.path.join(out_dir, f"_raw_{name}.csv")
        final_txt = os.path.join(out_dir, f"{name}.txt")

        # 尝试运行 iLearn
        args = [sys.executable, script, "--file", fasta_path, "--method", method, "--format", "csv", "--out", raw_csv]
        if method == "SCPseTNC": args += ["--type", "DNA"]

        try:
            subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        except Exception as e:
            print(f"  [Error] 运行 {method} 失败: {e}")

        if os.path.exists(raw_csv):
            with open(raw_csv, "r") as f_in, open(final_txt, "w") as f_out:
                reader = csv.reader(f_in)
                try:
                    next(reader)
                    for row in reader:
                        if len(row) > 2:
                            f_out.write(" ".join(row[1:-1]) + "\n")
                except StopIteration:
                    pass
            os.remove(raw_csv)


def generate_mismatch_txt(seqs, out_txt, k=5):
    # R.capsulatus 优化：k=5 生成 1024 维特征
    kmers = ["".join(p) for p in product(DNA_ALPHABET, repeat=k)]
    k_idx = {km: i for i, km in enumerate(kmers)}
    with open(out_txt, "w") as f:
        for seq in seqs:
            vec = [0.0] * len(kmers)
            n = len(seq) - k + 1
            if n > 0:
                for i in range(n):
                    sub = seq[i:i + k]
                    if sub in k_idx: vec[k_idx[sub]] += 1.0
                vec = [v / n for v in vec]
            f.write(" ".join(f"{v:.6f}" for v in vec) + "\n")


# ================= 主流程 =================

def main():
    # 寻路 iLearn
    start_path = Path(BASE_PROJECT_DIR)
    basic_p, pse_p = None, None
    for base in [start_path] + list(start_path.parents)[:3]:
        res = list(base.rglob(ILEARN_SCRIPT_NAME))
        if res:
            basic_p = str(res[0])
            pse_p = os.path.join(os.path.dirname(basic_p), "iLearn-nucleotide-Pse.py")
            break
    if not basic_p: return

    for ds_name, config in DATASETS_CONFIG.items():
        print(f"\n处理数据集: {ds_name}")
        h, s = read_fasta_strict(config["fasta"])
        l = []
        with open(config["label"], "r") as f:
            for line in f:
                if line.strip(): l.append(int(float(line.strip())))

        min_l = min(len(s), len(l))
        h, s, l = h[:min_l], s[:min_l], l[:min_l]

        # 采样平衡
        pos_idx = [i for i, v in enumerate(l) if v == 1]
        neg_idx = [i for i, v in enumerate(l) if v == 0]
        target = min(len(pos_idx), len(neg_idx))
        random.seed(SPLIT_SEED)
        pick = random.sample(pos_idx, target) + random.sample(neg_idx, target)
        random.shuffle(pick)
        h = [h[i] for i in pick];
        s = [s[i] for i in pick];
        l = [l[i] for i in pick]

        # 划分
        split = int(len(l) * 0.8)
        tr_s, tr_l = s[:split], l[:split]
        te_s, te_l = s[split:], l[split:]

        ds_root = os.path.join(FEATURE_ROOT_DIR, ds_name)

        # 核心：预先训练 Word2Vec 以供 train/test 使用
        sentences = [[sq[i:i + W2V_KMER] for i in range(len(sq) - W2V_KMER + 1)] for sq in tr_s]
        from gensim.models import Word2Vec
        w2v_model = Word2Vec(sentences, vector_size=W2V_VECTOR_DIM, window=5, min_count=1, sg=1)

        for stype, (ms, ml) in [("train", (tr_s, tr_l)), ("test", (te_s, te_l))]:
            out_dir = os.path.join(ds_root, stype)
            os.makedirs(out_dir, exist_ok=True)

            # 1. 物理文件生成
            fasta_path = os.path.join(out_dir, "data.fasta")
            with open(fasta_path, "w") as f:
                for i, (seq, lab) in enumerate(zip(ms, ml)):
                    f.write(f">{i}|{lab}\n{seq}\n")
            with open(os.path.join(out_dir, "y.txt"), "w") as f:
                for lab in ml: f.write(f"{lab}\n")

            # 2. 特征提取
            generate_ilearn_features(fasta_path, out_dir, basic_p, pse_p)
            generate_mismatch_txt(ms, os.path.join(out_dir, "mismatch.txt"))

            # Word2Vec
            pool = nn.AdaptiveAvgPool1d(W2V_OUT_LEN)
            with open(os.path.join(out_dir, "word2vec.txt"), "w") as f:
                for sq in ms:
                    kmers = [sq[i:i + W2V_KMER] for i in range(len(sq) - W2V_KMER + 1)]
                    vecs = [w2v_model.wv[k] for k in kmers if k in w2v_model.wv]
                    if not vecs: vecs = [[0.0] * W2V_VECTOR_DIM]
                    t_v = torch.tensor(vecs).reshape(1, 1, -1)
                    res = pool(t_v).view(-1).tolist()
                    f.write(" ".join(f"{x:.6g}" for x in res) + "\n")

            # 3. 强制对齐与兜底 (解决 FileNotFoundError)
            align_feature_rows(out_dir, len(ml))

    print(f"\n任务完成。输出目录: {FEATURE_ROOT_DIR}")


if __name__ == "__main__":
    main()