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
DATA_CELL_DIR = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/data/mouse nonTATA"
# 如果没有 train/test，则会用这个 seed 和 7:1 自动拆分
SPLIT_SEED = 10
TRAIN_TO_TEST = (7, 1)  # 7:1


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


def write_labeled_fasta(records, labels, out_path, tag):
    """
    写带标签的 FASTA（iLearn 需要 label 字段）：
      >header|label|training/testing
      SEQ...
    """
    if len(records) != len(labels):
        raise ValueError(f"FASTA 条目数({len(records)}) != 标签数({len(labels)})，请检查是否一一对应")

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
    这里统一改成：label,feat...（保留表头）
    """
    with open(in_csv, "r", encoding="utf-8", errors="ignore", newline="") as fin, \
         open(out_csv, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        rows = list(reader)
        if not rows:
            raise RuntimeError(f"iLearn 输出为空：{in_csv}")

        header = rows[0]
        if len(header) < 2:
            raise RuntimeError(f"iLearn 输出格式异常：{in_csv}")

        # 去掉第一列 '#'
        writer.writerow(header[1:])  # ['label', feat...]

        for row in rows[1:]:
            if not row:
                continue
            if len(row) < 2:
                continue
            # 去掉第一列 name
            writer.writerow(row[1:])


def generate_by_ilearn_basic(python_exe, ilearn_basic_py, fasta_in, method, out_csv_tmp, out_csv_final):
    run_cmd([
        python_exe, ilearn_basic_py,
        "--file", fasta_in,
        "--method", method,
        "--format", "csv",
        "--out", out_csv_tmp
    ])
    sanitize_ilearn_csv(out_csv_tmp, out_csv_final)
    os.remove(out_csv_tmp)


def generate_by_ilearn_pse(python_exe, ilearn_pse_py, fasta_in, method, out_csv_tmp, out_csv_final):
    run_cmd([
        python_exe, ilearn_pse_py,
        "--file", fasta_in,
        "--method", method,
        "--type", "DNA",
        "--format", "csv",
        "--out", out_csv_tmp
    ])
    sanitize_ilearn_csv(out_csv_tmp, out_csv_final)
    os.remove(out_csv_tmp)


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
                    res.add(s[:i] + b + s[i + 1:])

    if m >= 2:
        for i in range(L):
            for j in range(i + 1, L):
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
    if len(records) != len(labels):
        raise ValueError("records 与 labels 数量不一致")

    kmers = build_all_kmers(k)
    idx = {km: i for i, km in enumerate(kmers)}
    header = ["label"] + [f"{km}_mm{m}" for km in kmers]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for (name, seq), y in zip(records, labels):
            seq = seq.upper()
            n = len(seq) - k + 1
            vec = [0.0] * len(kmers)

            if n > 0:
                for p in range(n):
                    win = seq[p:p + k]
                    if any(ch not in "ACGT" for ch in win):
                        continue
                    for nb in neighbors_with_mismatches(win, m):
                        vec[idx[nb]] += 1.0

                denom = float(n)
                vec = [v / denom for v in vec]

            writer.writerow([int(y)] + vec)


def infer_paths(data_cell_dir):
    """
    输入：.../iPro-WAEL-main/iPro-WAEL-main/data/NHEK
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
    """判断是否已经有 data/<cell>/train 和 data/<cell>/test 的完整文件"""
    train_dir = os.path.join(ipro_root, "data", cell, "train")
    test_dir = os.path.join(ipro_root, "data", cell, "test")
    ok_train = os.path.exists(os.path.join(train_dir, "data.fasta")) and os.path.exists(os.path.join(train_dir, "y_train.txt"))
    ok_test = os.path.exists(os.path.join(test_dir, "data.fasta")) and os.path.exists(os.path.join(test_dir, "y_test.txt"))
    return ok_train and ok_test


def find_unsplit_files(data_cell_dir):
    """
    没有 train/test 时，从 data/<cell>/ 下寻找：
      data.fasta
      y.txt 或 y_train.txt 或 y_label.txt（按常见名字猜）
    """
    fasta = os.path.join(data_cell_dir, "data.fasta")
    if not os.path.exists(fasta):
        raise FileNotFoundError(f"未找到未划分的 data.fasta：{fasta}")

    candidates = [
        os.path.join(data_cell_dir, "y.txt"),
        os.path.join(data_cell_dir, "y_train.txt"),
        os.path.join(data_cell_dir, "y_label.txt"),
        os.path.join(data_cell_dir, "labels.txt"),
    ]
    y_path = None
    for p in candidates:
        if os.path.exists(p):
            y_path = p
            break
    if y_path is None:
        raise FileNotFoundError(
            f"未找到标签文件。请把标签文件命名为 y.txt / y_train.txt / y_label.txt / labels.txt 之一，并放到：{data_cell_dir}"
        )
    return fasta, y_path


def stratified_split_indices(labels, test_frac, seed):
    """按 label 分层拆分"""
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


def auto_make_train_test(ipro_root, cell, data_cell_dir, seed=SPLIT_SEED, ratio=TRAIN_TO_TEST):
    """
    如果没有 train/test，则自动按 7:1 生成：
      data/<cell>/train/data.fasta + y_train.txt
      data/<cell>/test/data.fasta  + y_test.txt
    """
    fasta_path, y_path = find_unsplit_files(data_cell_dir)

    records = read_fasta_records(fasta_path)
    labels = read_labels(y_path)

    if len(records) != len(labels):
        raise ValueError(f"未划分数据：FASTA({len(records)}) != labels({len(labels)})")

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

    print(f"[AUTO-SPLIT] 已生成 train/test：train={len(train_records)}, test={len(test_records)}, seed={seed}, ratio={train_r}:{test_r}")


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

    # 1) CKSNAP
    tmp = os.path.join(out_dir, "_cksnap_tmp.csv")
    final = os.path.join(out_dir, "cksnap.csv")
    generate_by_ilearn_basic(py, ilearn_basic, labeled_fasta, "CKSNAP", tmp, final)
    print(f"[OK] saved -> {final}")

    # 2) RCKmer
    tmp = os.path.join(out_dir, "_rckmer_tmp.csv")
    final = os.path.join(out_dir, "rckmer.csv")
    generate_by_ilearn_basic(py, ilearn_basic, labeled_fasta, "RCKmer", tmp, final)
    print(f"[OK] saved -> {final}")

    # 3) 用 TNC 代替 TPCP（文件名保持 tpcp.csv）
    tmp = os.path.join(out_dir, "_tnc_tmp.csv")
    final = os.path.join(out_dir, "tpcp.csv")
    generate_by_ilearn_basic(py, ilearn_basic, labeled_fasta, "TNC", tmp, final)
    print(f"[OK] saved -> {final}   (注意：这里用 TNC 替代 TPCP)")

    # 4) SCPseTNC -> psetnc.csv
    tmp = os.path.join(out_dir, "_scpseTNC_tmp.csv")
    final = os.path.join(out_dir, "psetnc.csv")
    generate_by_ilearn_pse(py, ilearn_pse, labeled_fasta, "SCPseTNC", tmp, final)
    print(f"[OK] saved -> {final}   (SCPseTNC)")

    # 5) mismatch profile（纯 Python）
    final = os.path.join(out_dir, "mismatch.csv")
    generate_mismatch_profile_csv(records, labels, final, k=5, m=1)
    print(f"[OK] saved -> {final}   (mismatch profile k=5,m=1)")


def main():
    data_cell_dir = os.path.abspath(DATA_CELL_DIR)
    if not os.path.isdir(data_cell_dir):
        raise FileNotFoundError(f"找不到 data_cell_dir：{data_cell_dir}")

    ipro_root, cell, ilearn_basic, ilearn_pse = infer_paths(data_cell_dir)

    os.makedirs(FEATURE_ROOT, exist_ok=True)

    print(f"[INFO] ipro_root={ipro_root}")
    print(f"[INFO] cell={cell}")
    print(f"[INFO] ilearn_basic={ilearn_basic}")
    print(f"[INFO] ilearn_pse={ilearn_pse}")
    print(f"[INFO] FEATURE_ROOT={FEATURE_ROOT}")

    # ✅ 新增判断：没有 train/test 就自动 7:1 划分
    if not has_train_test_split(ipro_root, cell):
        print("[INFO] 未检测到已划分的 train/test，开始自动按 7:1 分层划分生成 train/test ...")
        auto_make_train_test(ipro_root, cell, data_cell_dir, seed=SPLIT_SEED, ratio=TRAIN_TO_TEST)
    else:
        print("[INFO] 检测到已存在 train/test，直接使用现有划分。")

    # 生成特征
    process_split(ipro_root, cell, ilearn_basic, ilearn_pse, "train")
    process_split(ipro_root, cell, ilearn_basic, ilearn_pse, "test")

    print("\n[DONE] train + test 特征已全部生成完成！")
    print(f"[DONE] 输出根目录：{FEATURE_ROOT}")


if __name__ == "__main__":
    main()
