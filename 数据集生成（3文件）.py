#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import subprocess
import random
from itertools import product

FEATURE_ROOT = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/feature"

DNA_ALPHABET = ["A", "C", "G", "T"]

# ========= 你只需要改这里：数据集目录（cell） =========
DATA_CELL_DIR = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/data/B.subtilis"

# ========= 3个文件（你给的）=========
# data.fasta 可选（没有标签文件时它不够用）
UNSPLIT_FASTA = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/data/B.subtilis/data.fasta"
NEG_FASTA = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/data/B.subtilis/Bacillus_non_prom.fasta"  # label=0
POS_FASTA = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/data/B.subtilis/Bacillus_prom.fasta"      # label=1

# 如果你是 “data.fasta + y.txt” 模式，就填这个（没有就留 None）
UNSPLIT_Y = None  # 例如: "/.../data/mouse TATA/y.txt"

# 自动拆分参数：7:1
SPLIT_SEED = 10
TRAIN_TO_TEST = (7, 1)  # 7:1

# 需要生成的特征（注意：iLearn basic 不支持 Mismatch/PseTNC/TPCP，我们保持之前策略）
# - cksnap: iLearn basic CKSNAP
# - rckmer: iLearn basic RCKmer
# - tpcp: 用 TNC 替代，文件名仍叫 tpcp.csv
# - psetnc: 用 iLearn Pse 脚本的 SCPseTNC
# - mismatch: 用纯Python mismatch profile(k=5,m=1)
METHODS = ["cksnap", "rckmer", "tpcp", "psetnc", "mismatch"]


def read_fasta_records(fasta_path):
    """读取 FASTA 为 records: [(header_without_>, seq_string), ...]"""
    records = []
    header = None
    seq_parts = []
    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if header is not None:
            records.append((header, "".join(seq_parts)))
    return records


def write_fasta_records(records, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for h, seq in records:
            w.write(f">{h}\n")
            w.write(seq + "\n")


def read_labels(y_path):
    """读取 y 文件：每行一个 label，可为 0/1 或 1.000000e+00"""
    labels = []
    with open(y_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = float(line)
            labels.append(1 if v >= 0.5 else 0)
    return labels


def write_labels(labels, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for y in labels:
            w.write(f"{int(y)}\n")


def infer_paths(data_cell_dir):
    """
    输入：.../iPro-WAEL-main/iPro-WAEL-main/data/<cell>
    推断 iLearn 脚本位置：
      .../iPro-WAEL-main/iLearn/iLearn-nucleotide-basic.py
      .../iPro-WAEL-main/iLearn/iLearn-nucleotide-Pse.py
    """
    data_cell_dir = os.path.abspath(data_cell_dir)
    cell = os.path.basename(data_cell_dir)
    data_dir = os.path.dirname(data_cell_dir)          # .../data
    ipro_root = os.path.dirname(data_dir)              # .../iPro-WAEL-main/iPro-WAEL-main
    outer = os.path.dirname(ipro_root)                 # .../iPro-WAEL-main

    ilearn_dir = os.path.join(outer, "iLearn")
    ilearn_basic = os.path.join(ilearn_dir, "iLearn-nucleotide-basic.py")
    ilearn_pse = os.path.join(ilearn_dir, "iLearn-nucleotide-Pse.py")
    return ipro_root, cell, ilearn_basic, ilearn_pse


def has_train_test_split(ipro_root, cell):
    train_dir = os.path.join(ipro_root, "data", cell, "train")
    test_dir = os.path.join(ipro_root, "data", cell, "test")
    ok_train = os.path.exists(os.path.join(train_dir, "data.fasta")) and os.path.exists(os.path.join(train_dir, "y_train.txt"))
    ok_test = os.path.exists(os.path.join(test_dir, "data.fasta")) and os.path.exists(os.path.join(test_dir, "y_test.txt"))
    return ok_train and ok_test


def stratified_split_indices(labels, test_frac, seed):
    rnd = random.Random(seed)
    idx0 = [i for i, y in enumerate(labels) if y == 0]
    idx1 = [i for i, y in enumerate(labels) if y == 1]
    rnd.shuffle(idx0)
    rnd.shuffle(idx1)

    def split_one(idxs):
        n = len(idxs)
        n_test = int(round(n * test_frac))
        if n >= 2:
            n_test = max(1, n_test)
        else:
            n_test = 0
        test = idxs[:n_test]
        train = idxs[n_test:]
        return train, test

    tr0, te0 = split_one(idx0)
    tr1, te1 = split_one(idx1)

    train_idx = tr0 + tr1
    test_idx = te0 + te1
    rnd.shuffle(train_idx)
    rnd.shuffle(test_idx)
    return train_idx, test_idx


def auto_make_train_test_from_pos_neg(ipro_root, cell, pos_fasta, neg_fasta, seed, ratio):
    pos_records = read_fasta_records(pos_fasta)
    neg_records = read_fasta_records(neg_fasta)

    records = pos_records + neg_records
    labels = [1] * len(pos_records) + [0] * len(neg_records)

    train_r, test_r = ratio
    test_frac = test_r / float(train_r + test_r)
    train_idx, test_idx = stratified_split_indices(labels, test_frac=test_frac, seed=seed)

    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    train_dir = os.path.join(ipro_root, "data", cell, "train")
    test_dir = os.path.join(ipro_root, "data", cell, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    write_fasta_records(train_records, os.path.join(train_dir, "data.fasta"))
    write_fasta_records(test_records, os.path.join(test_dir, "data.fasta"))
    write_labels(y_train, os.path.join(train_dir, "y_train.txt"))
    write_labels(y_test, os.path.join(test_dir, "y_test.txt"))

    print(f"[AUTO-SPLIT pos/neg] train={len(train_records)}, test={len(test_records)}, seed={seed}, ratio={train_r}:{test_r}")


def auto_make_train_test_from_fasta_y(ipro_root, cell, fasta_path, y_path, seed, ratio):
    records = read_fasta_records(fasta_path)
    labels = read_labels(y_path)
    if len(records) != len(labels):
        raise ValueError(f"FASTA({len(records)}) != labels({len(labels)})")

    train_r, test_r = ratio
    test_frac = test_r / float(train_r + test_r)
    train_idx, test_idx = stratified_split_indices(labels, test_frac=test_frac, seed=seed)

    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    train_dir = os.path.join(ipro_root, "data", cell, "train")
    test_dir = os.path.join(ipro_root, "data", cell, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    write_fasta_records(train_records, os.path.join(train_dir, "data.fasta"))
    write_fasta_records(test_records, os.path.join(test_dir, "data.fasta"))
    write_labels(y_train, os.path.join(train_dir, "y_train.txt"))
    write_labels(y_test, os.path.join(test_dir, "y_test.txt"))

    print(f"[AUTO-SPLIT fasta+y] train={len(train_records)}, test={len(test_records)}, seed={seed}, ratio={train_r}:{test_r}")


def write_labeled_fasta(records, labels, out_path, tag):
    if len(records) != len(labels):
        raise ValueError(f"FASTA 条目数({len(records)}) != 标签数({len(labels)})")
    with open(out_path, "w", encoding="utf-8") as w:
        for (h, seq), y in zip(records, labels):
            w.write(f">{h}|{int(y)}|{tag}\n")
            w.write(seq + "\n")


def run_cmd(cmd):
    print("\n[RUN] " + " ".join([f'"{c}"' if " " in c else c for c in cmd]))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.stdout.strip():
        print("[output]\n" + r.stdout)
    if r.returncode != 0:
        raise RuntimeError(f"命令运行失败，exit_code={r.returncode}")


def sanitize_ilearn_csv(in_csv, out_csv):
    """
    iLearn csv 默认是：#,label,feat...
    行是：name,label,feat...
    输出：label,feat...（保留表头）
    """
    with open(in_csv, "r", encoding="utf-8", errors="ignore", newline="") as fin, \
         open(out_csv, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        rows = list(reader)
        if not rows:
            raise RuntimeError(f"iLearn 输出为空：{in_csv}")
        header = rows[0]
        writer.writerow(header[1:])      # 去掉 '#'
        for row in rows[1:]:
            if not row:
                continue
            writer.writerow(row[1:])     # 去掉 name


def generate_by_ilearn_basic(python_exe, ilearn_basic_py, fasta_in, method, out_csv_final):
    tmp = out_csv_final + ".raw_tmp.csv"
    run_cmd([
        python_exe, ilearn_basic_py,
        "--file", fasta_in,
        "--method", method,
        "--format", "csv",
        "--out", tmp
    ])
    sanitize_ilearn_csv(tmp, out_csv_final)
    os.remove(tmp)


def generate_by_ilearn_pse(python_exe, ilearn_pse_py, fasta_in, method, out_csv_final):
    tmp = out_csv_final + ".raw_tmp.csv"
    run_cmd([
        python_exe, ilearn_pse_py,
        "--file", fasta_in,
        "--method", method,
        "--type", "DNA",
        "--format", "csv",
        "--out", tmp
    ])
    sanitize_ilearn_csv(tmp, out_csv_final)
    os.remove(tmp)


def build_all_kmers(k):
    return ["".join(p) for p in product(DNA_ALPHABET, repeat=k)]


def neighbors_with_mismatches(s, m):
    if m == 0:
        return [s]
    res = {s}
    L = len(s)
    if m >= 1:
        for i in range(L):
            for b in DNA_ALPHABET:
                if b != s[i]:
                    res.add(s[:i] + b + s[i+1:])
    if m >= 2:
        for i in range(L):
            for j in range(i+1, L):
                for bi in DNA_ALPHABET:
                    if bi == s[i]:
                        continue
                    for bj in DNA_ALPHABET:
                        if bj == s[j]:
                            continue
                        t = list(s)
                        t[i] = bi
                        t[j] = bj
                        res.add("".join(t))
    return list(res)


def generate_mismatch_profile_csv(records, labels, out_csv, k=5, m=1):
    kmers = build_all_kmers(k)
    idx = {km: i for i, km in enumerate(kmers)}
    header = ["label"] + [f"{km}_mm{m}" for km in kmers]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for (_, seq), y in zip(records, labels):
            seq = seq.upper()
            n = len(seq) - k + 1
            vec = [0.0] * len(kmers)
            if n > 0:
                for p in range(n):
                    win = seq[p:p+k]
                    if any(ch not in "ACGT" for ch in win):
                        continue
                    for nb in neighbors_with_mismatches(win, m):
                        vec[idx[nb]] += 1.0
                denom = float(n)
                vec = [v / denom for v in vec]
            writer.writerow([int(y)] + vec)


def process_split(ipro_root, cell, ilearn_basic, ilearn_pse, split):
    assert split in ("train", "test")
    data_split_dir = os.path.join(ipro_root, "data", cell, split)
    fasta_path = os.path.join(data_split_dir, "data.fasta")
    y_path = os.path.join(data_split_dir, "y_train.txt" if split == "train" else "y_test.txt")

    out_dir = os.path.join(FEATURE_ROOT, cell, split)
    os.makedirs(out_dir, exist_ok=True)

    tag = "training" if split == "train" else "testing"
    labeled_fasta = os.path.join("/tmp", f"{cell}_{split}_labeled.fasta")

    print("\n==============================")
    print(f"[SPLIT] {cell}/{split}")
    print(f"[INFO] fasta={fasta_path}")
    print(f"[INFO] y={y_path}")
    print(f"[INFO] out_dir={out_dir}")
    print("==============================")

    records = read_fasta_records(fasta_path)
    labels = read_labels(y_path)
    print(f"[INFO] fasta_records={len(records)}, labels={len(labels)}")

    write_labeled_fasta(records, labels, labeled_fasta, tag=tag)
    print(f"[OK] labeled fasta -> {labeled_fasta}")

    py = sys.executable

    # cksnap
    if "cksnap" in METHODS:
        generate_by_ilearn_basic(py, ilearn_basic, labeled_fasta, "CKSNAP", os.path.join(out_dir, "cksnap.csv"))
        print(f"[OK] saved -> {os.path.join(out_dir, 'cksnap.csv')}")

    # rckmer
    if "rckmer" in METHODS:
        generate_by_ilearn_basic(py, ilearn_basic, labeled_fasta, "RCKmer", os.path.join(out_dir, "rckmer.csv"))
        print(f"[OK] saved -> {os.path.join(out_dir, 'rckmer.csv')}")

    # tpcp (用 TNC 替代)
    if "tpcp" in METHODS:
        generate_by_ilearn_basic(py, ilearn_basic, labeled_fasta, "TNC", os.path.join(out_dir, "tpcp.csv"))
        print(f"[OK] saved -> {os.path.join(out_dir, 'tpcp.csv')}   (TNC 替代 TPCP)")

    # psetnc (SCPseTNC)
    if "psetnc" in METHODS:
        generate_by_ilearn_pse(py, ilearn_pse, labeled_fasta, "SCPseTNC", os.path.join(out_dir, "psetnc.csv"))
        print(f"[OK] saved -> {os.path.join(out_dir, 'psetnc.csv')}   (SCPseTNC)")

    # mismatch (纯Python)
    if "mismatch" in METHODS:
        generate_mismatch_profile_csv(records, labels, os.path.join(out_dir, "mismatch.csv"), k=5, m=1)
        print(f"[OK] saved -> {os.path.join(out_dir, 'mismatch.csv')}   (mismatch k=5,m=1)")


def main():
    data_cell_dir = os.path.abspath(DATA_CELL_DIR)
    if not os.path.isdir(data_cell_dir):
        raise FileNotFoundError(f"找不到 DATA_CELL_DIR：{data_cell_dir}")

    ipro_root, cell, ilearn_basic, ilearn_pse = infer_paths(data_cell_dir)

    os.makedirs(FEATURE_ROOT, exist_ok=True)

    print(f"[INFO] ipro_root={ipro_root}")
    print(f"[INFO] cell={cell}")
    print(f"[INFO] ilearn_basic={ilearn_basic}")
    print(f"[INFO] ilearn_pse={ilearn_pse}")
    print(f"[INFO] FEATURE_ROOT={FEATURE_ROOT}")

    # 1) 如果已存在 train/test，就直接用
    if has_train_test_split(ipro_root, cell):
        print("[INFO] 检测到已存在 train/test，直接使用现有划分。")
    else:
        # 2) 否则：优先使用 pos/neg 两个 fasta（你现在的三文件就是这种）
        if POS_FASTA and NEG_FASTA and os.path.exists(POS_FASTA) and os.path.exists(NEG_FASTA):
            print("[INFO] 未检测到 train/test，使用 POS/NEG FASTA 自动生成标签并按 7:1 分层划分...")
            auto_make_train_test_from_pos_neg(ipro_root, cell, POS_FASTA, NEG_FASTA, seed=SPLIT_SEED, ratio=TRAIN_TO_TEST)
        # 3) 如果你提供了 data.fasta + y.txt，也可以走这个分支
        elif UNSPLIT_FASTA and UNSPLIT_Y and os.path.exists(UNSPLIT_FASTA) and os.path.exists(UNSPLIT_Y):
            print("[INFO] 未检测到 train/test，使用 data.fasta + y 自动按 7:1 分层划分...")
            auto_make_train_test_from_fasta_y(ipro_root, cell, UNSPLIT_FASTA, UNSPLIT_Y, seed=SPLIT_SEED, ratio=TRAIN_TO_TEST)
        else:
            raise RuntimeError(
                "没有 train/test，也没有可用的 (POS_FASTA + NEG_FASTA) 或 (data.fasta + y.txt)。\n"
                "请检查 POS_FASTA/NEG_FASTA 路径是否正确，或提供 UNSPLIT_Y。"
            )

    # 生成 train/test 特征
    process_split(ipro_root, cell, ilearn_basic, ilearn_pse, "train")
    process_split(ipro_root, cell, ilearn_basic, ilearn_pse, "test")

    print("\n[DONE] train + test 特征已全部生成完成！")
    print(f"[DONE] 输出根目录：{FEATURE_ROOT}")


if __name__ == "__main__":
    main()
