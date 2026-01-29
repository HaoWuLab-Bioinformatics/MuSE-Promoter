# -*- coding: utf-8 -*-
"""
【完整版】Mouse TATA/nonTATA 数据集构建 + 特征生成
针对 CNNPromoterData-master 下的五个 .fa 文件进行处理
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

# 项目根目录
BASE_PROJECT_DIR = os.path.join(SCRIPT_DIR, "iPro-WAEL-main")
FEATURE_ROOT_DIR = os.path.join(BASE_PROJECT_DIR, "feature")

# iLearn 脚本位置
ILEARN_SCRIPT_NAME = "iLearn-nucleotide-basic.py"

# 数据原始路径
MOUSE_DATA_DIR = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/CNNPromoterData-master"

# ====== 任务配置 ======
# mouse_TATA: 正样本(Mouse_tata.fa) + 负样本(Mouse_nonprom.fa)
# mouse_nonTATA: 正样本(Mouse_non_tata.fa) + 负样本(Mouse_nonprom.fa)
DATASETS_CONFIG = {
    "mouse_TATA": {
        "pos": os.path.join(MOUSE_DATA_DIR, "Mouse_tata.fa"),
        "neg": os.path.join(MOUSE_DATA_DIR, "Mouse_nonprom.fa"),
    },
    "mouse_nonTATA": {
        "pos": os.path.join(MOUSE_DATA_DIR, "Mouse_non_tata.fa"),
        "neg": os.path.join(MOUSE_DATA_DIR, "Mouse_nonprom.fa"),
    }
}

# 划分参数
SPLIT_SEED = 10
TRAIN_TO_TEST = (7, 1)  # 约 87% 训练, 13% 测试

# DNA及Word2Vec配置
DNA_ALPHABET = ["A", "C", "G", "T"]
W2V_KMER = 5
W2V_VECTOR_DIM = 100
W2V_OUT_LEN = 8000


# ================= 数据读取与准备 =================

def read_fasta_simple(path, label):
    """读取FASTA并赋予统一标签"""
    headers, seqs, labels = [], [], []
    if not os.path.exists(path):
        print(f"[Warn] 文件不存在: {path}")
        return headers, seqs, labels

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        curr_seq = []
        curr_head = None
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if curr_seq:
                    seqs.append("".join(curr_seq).upper())
                    labels.append(label)
                curr_head = line[1:]
                headers.append(curr_head)
                curr_seq = []
            else:
                curr_seq.append(line)
        if curr_seq:
            seqs.append("".join(curr_seq).upper())
            labels.append(label)
    return headers, seqs, labels


def prepare_dataset(name, config):
    print(f"\n--- 准备数据集: {name} ---")

    # 读取正负样本
    pos_h, pos_s, pos_l = read_fasta_simple(config["pos"], 1)
    neg_h, neg_s, neg_l = read_fasta_simple(config["neg"], 0)

    if not pos_s or not neg_s:
        print(f"[ERROR] {name} 读取数据失败，请检查路径。")
        return [], [], []

    print(f"    [Load] 原始正样本: {len(pos_s)}, 原始负样本: {len(neg_s)}")

    # 类别平衡 (下采样)
    target = min(len(pos_s), len(neg_s))
    random.seed(SPLIT_SEED)

    p_idx = random.sample(range(len(pos_s)), target)
    n_idx = random.sample(range(len(neg_s)), target)

    headers = [pos_h[i] for i in p_idx] + [neg_h[i] for i in n_idx]
    seqs = [pos_s[i] for i in p_idx] + [neg_s[i] for i in n_idx]
    labels = [pos_l[i] for i in p_idx] + [neg_l[i] for i in n_idx]

    # 打乱
    combined = list(zip(headers, seqs, labels))
    random.shuffle(combined)
    out_h, out_s, out_l = zip(*combined)

    print(f"    [Stat] 平衡后总数: {len(out_l)} (正负各 {target})")
    return list(out_h), list(out_s), list(out_l)


# ================= 辅助工具 =================

def _count_lines(path):
    if not os.path.exists(path): return 0
    with open(path, "r", errors="ignore") as f:
        return sum(1 for _ in f)


def _pad_or_trim_txt(path, target_lines):
    if not os.path.exists(path): return False
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
    return True


def align_feature_rows(out_dir):
    y_path = os.path.join(out_dir, "y.txt")
    if not os.path.exists(y_path): return
    target = _count_lines(y_path)
    feats = ["cksnap.txt", "mismatch.txt", "rckmer.txt", "psetnc.txt", "tpcp.txt", "word2vec.txt"]
    for fn in feats:
        fp = os.path.join(out_dir, fn)
        if os.path.exists(fp):
            _pad_or_trim_txt(fp, target)


def write_fasta_and_y(headers, seqs, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fasta_path = os.path.join(out_dir, "data.fasta")
    y_path = os.path.join(out_dir, "y.txt")
    with open(fasta_path, "w", encoding="utf-8") as wf, open(y_path, "w", encoding="utf-8") as wy:
        for i, (h, s, l) in enumerate(zip(headers, seqs, labels)):
            wf.write(f">{i}|{int(l)}\n{s}\n")
            wy.write(f"{int(l)}\n")
    return fasta_path


def stratified_split_data(headers, seqs, labels, ratio=TRAIN_TO_TEST):
    data = list(zip(headers, seqs, labels))
    random.seed(SPLIT_SEED)
    random.shuffle(data)
    pos = [d for d in data if d[2] == 1]
    neg = [d for d in data if d[2] == 0]
    total_r = sum(ratio)
    n_pos_test = int(len(pos) * (ratio[1] / total_r))
    n_neg_test = int(len(neg) * (ratio[1] / total_r))
    te = pos[:n_pos_test] + neg[:n_neg_test]
    tr = pos[n_pos_test:] + neg[n_neg_test:]
    random.shuffle(te)
    random.shuffle(tr)

    def unzip(ds):
        if not ds: return [], [], []
        h, s, l = zip(*ds)
        return list(h), list(s), list(l)

    return unzip(tr), unzip(te)


# ================= 特征引擎 =================

def find_ilearn_paths(start_dir, script_name):
    print(f"[寻路] 搜索 {script_name} ...")
    start_path = Path(start_dir)
    search_bases = [start_path] + list(start_path.parents)[:3]
    for base in search_bases:
        try:
            results = list(base.rglob(script_name))
            if results:
                found_basic = str(results[0])
                found_pse = os.path.join(os.path.dirname(found_basic), "iLearn-nucleotide-Pse.py")
                return found_basic, found_pse
        except:
            continue
    return None, None


def generate_ilearn_features(data_dir, headers, seqs, labels, basic_script, pse_script):
    tmp_fasta = os.path.join(data_dir, "temp_ilearn.fasta")
    with open(tmp_fasta, "w") as f:
        for i, (h, s, y) in enumerate(zip(headers, seqs, labels)):
            f.write(f">S{i}|{int(y)}\n{s}\n")

    tasks = [("CKSNAP", basic_script, "cksnap"), ("RCKmer", basic_script, "rckmer"),
             ("TNC", basic_script, "tpcp"), ("SCPseTNC", pse_script, "psetnc")]

    for method, script, name in tasks:
        out_csv = os.path.join(data_dir, f"raw_{name}.csv")
        args = [sys.executable, script, "--file", tmp_fasta, "--method", method, "--format", "csv", "--out", out_csv]
        if method == "SCPseTNC": args += ["--type", "DNA"]

        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        txt_path = os.path.join(data_dir, f"{name}.txt")
        if os.path.exists(out_csv):
            with open(out_csv, "r") as f_in, open(txt_path, "w") as f_out:
                reader = csv.reader(f_in)
                header_row = next(reader)
                for row in reader:
                    # 去除 ID 列和标签列
                    f_out.write(" ".join(row[1:-1]) + "\n")
            os.remove(out_csv)
            print(f"    [OK] 生成: {name}.txt")
    if os.path.exists(tmp_fasta): os.remove(tmp_fasta)


def generate_mismatch_txt(seqs, out_txt, k=5, m=1):
    kmers = ["".join(p) for p in product(DNA_ALPHABET, repeat=k)]
    k_idx = {km: i for i, km in enumerate(kmers)}
    with open(out_txt, "w") as f:
        for seq in seqs:
            vec = [0.0] * len(kmers)
            n = len(seq) - k + 1
            if n > 0:
                for i in range(n):
                    sub = seq[i:i + k]
                    if any(c not in DNA_ALPHABET for c in sub): continue
                    # 简化版：仅计算原位点
                    if sub in k_idx: vec[k_idx[sub]] += 1.0
                vec = [v / n for v in vec]
            f.write(" ".join(f"{v:.6f}" for v in vec) + "\n")
    print(f"    [OK] 生成: mismatch.txt")


def train_and_save_word2vec(train_seqs, index_path, w2v_path):
    from gensim.models import Word2Vec
    sentences = [[s[i:i + W2V_KMER] for i in range(len(s) - W2V_KMER + 1)] for s in train_seqs]
    model = Word2Vec(sentences, vector_size=W2V_VECTOR_DIM, window=5, min_count=1, sg=1)
    keys = list(model.wv.index_to_key)
    with open(index_path, "w") as f: f.write(" ".join(keys))
    with open(w2v_path, "w") as f:
        for k in keys: f.write(" ".join(map(str, model.wv[k])) + "\n")
    return keys, torch.tensor(model.wv.vectors)


def generate_w2v_txt(seqs, keys, w2v_mat, out_path):
    k_map = {k: i for i, k in enumerate(keys)}
    pool = nn.AdaptiveAvgPool1d(W2V_OUT_LEN)
    with open(out_path, "w") as f:
        for s in seqs:
            ids = [k_map.get(s[i:i + W2V_KMER], 0) for i in range(len(s) - W2V_KMER + 1)]
            if not ids: ids = [0]
            emb = w2v_mat[torch.tensor(ids)].reshape(1, 1, -1)
            out = pool(emb).view(-1).tolist()
            f.write(" ".join(f"{x:.6g}" for x in out) + "\n")
    print(f"    [OK] 生成: word2vec.txt")


# ================= 主程序 =================

def main():
    basic, pse = find_ilearn_paths(BASE_PROJECT_DIR, ILEARN_SCRIPT_NAME)
    if not basic: return

    for ds_name, config in DATASETS_CONFIG.items():
        print(f"\n########################################")
        print(f"### 正在处理任务: {ds_name}")

        all_h, all_s, all_l = prepare_dataset(ds_name, config)
        if not all_s: continue

        tr_data, te_data = stratified_split_data(all_h, all_s, all_l)

        ds_out_root = os.path.join(FEATURE_ROOT_DIR, ds_name)
        os.makedirs(ds_out_root, exist_ok=True)

        # Word2Vec 仅用训练集训练
        print(f"[W2V] 正在训练模型...")
        w2v_keys, w2v_mat = train_and_save_word2vec(tr_data[1],
                                                    os.path.join(ds_out_root, "w2v_idx.txt"),
                                                    os.path.join(ds_out_root, "w2v_vec.txt"))

        splits = {"train": tr_data, "test": te_data}
        for stype, (h, s, l) in splits.items():
            out_dir = os.path.join(ds_out_root, stype)
            print(f"\n    >>> {stype.upper()} 集: {len(l)} 样本 -> {out_dir}")

            write_fasta_and_y(h, s, l, out_dir)
            generate_ilearn_features(out_dir, h, s, l, basic, pse)
            generate_mismatch_txt(s, os.path.join(out_dir, "mismatch.txt"))
            generate_w2v_txt(s, w2v_keys, w2v_mat, os.path.join(out_dir, "word2vec.txt"))
            align_feature_rows(out_dir)

    print("\n=== 全部任务完成 ===")


if __name__ == "__main__":
    main()