#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import random
import subprocess
from itertools import product
from typing import List, Tuple

DNA_ALPHABET = ["A", "C", "G", "T"]

# ====== 你自己的路径配置（只改这里）======
PROJECT_ROOT = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main"
ILEARN_DIR   = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iLearn"
CELL_NAME    = "mouse nonTATA"

# 7:1 split
SEED = 10
TEST_FRACTION = 1 / 8

# mismatch profile 参数（“错配 k-mer”常用设置）
MISMATCH_K = 5
MISMATCH_M = 1

# PseTNC 方法（iLearn 的 Pse 脚本里常见：SCPseTNC / PCPseTNC）
PSETNC_METHOD = "SCPseTNC"

# TPCP：iLearn 没有真正 TPCP，这里默认用 TNC 代替输出为 tpcp.csv（可跑通）
USE_TNC_AS_TPCP = True


def read_fasta_records(fasta_path: str) -> List[Tuple[str, List[str]]]:
    """读取 FASTA，返回 [(header_line_with_>, [seq_lines...]), ...]"""
    records = []
    header = None
    seq_lines = []
    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, seq_lines))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if header is not None:
            records.append((header, seq_lines))
    return records


def write_fasta_records(out_path: str, records: List[Tuple[str, List[str]]]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for h, seq in records:
            w.write(h + "\n")
            for s in seq:
                w.write(s + "\n")


def run_ilearn(python_exe: str, script_path: str, fasta_in: str, method: str, out_csv: str, extra_args=None) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cmd = [python_exe, script_path, "--file", fasta_in, "--method", method, "--format", "csv", "--out", out_csv]
    if extra_args:
        cmd = [python_exe, script_path, "--file", fasta_in] + extra_args + ["--method", method, "--format", "csv", "--out", out_csv]

    print("\n[Run] " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    res = subprocess.run(cmd, cwd=os.path.dirname(script_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.stdout.strip():
        print("[iLearn output]\n" + res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"iLearn 运行失败：method={method}, exit_code={res.returncode}")
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        raise RuntimeError(f"iLearn 没有生成输出文件或文件为空：{out_csv}")


def convert_raw_keep_header_drop_namecol(raw_csv: str, out_csv: str) -> None:
    """
    保留表头，但删除第1列(样本名字符串)：
      表头：从第2列开始（label + features）
      数据：从第2列开始（label + features）
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(raw_csv, "r", encoding="utf-8", errors="ignore", newline="") as fin, \
         open(out_csv, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"raw csv 为空：{raw_csv}")
        if len(header) < 3:
            raise RuntimeError(f"raw csv 表头列数异常：{raw_csv}")

        writer.writerow(header[1:])  # drop first col (#)
        row_count = 0
        for row in reader:
            if not row or len(row) < 3:
                continue
            writer.writerow(row[1:])  # drop sample name col
            row_count += 1

    if row_count == 0:
        raise RuntimeError(f"转换后没有写入任何数据：{raw_csv}")
    print(f"[OK] keep header csv -> {out_csv} (rows={row_count})")


# ---------- mismatch k-mer（纯 Python） ----------
def build_all_kmers(k: int):
    return ["".join(p) for p in product(DNA_ALPHABET, repeat=k)]


def neighbors_with_mismatches(s: str, m: int):
    """生成 <=m mismatch 邻居（支持 m=0/1/2）"""
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


def generate_mismatch_profile_csv(fasta_path: str, out_csv: str, k=5, m=1):
    """
    mismatch k-mer profile:
    - 特征维度 4^k
    - 对每个窗口 k-mer，给所有 <=m mismatch 的 pattern 计数 +1
    - 用窗口数归一化
    - 输出 csv：第一列 label（这里没有标签，就输出 0 占位），后面是特征
      （注意：如果你有 y.txt，可以把 label 写进去；你当前脚本没带标签文件）
    """
    records = read_fasta_records(fasta_path)
    kmers = build_all_kmers(k)
    idx = {km: i for i, km in enumerate(kmers)}

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"{km}_mm{m}" for km in kmers])

        for header, seq_lines in records:
            seq = "".join(seq_lines).upper()
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

            # ⚠️ 这里 label 先写 0 占位。若你有 y 文件，我可以改成真实标签。
            writer.writerow([0] + vec)

    print(f"[OK] mismatch csv -> {out_csv} (k={k}, m={m})")


def main():
    input_fasta = os.path.join(PROJECT_ROOT, "data", CELL_NAME, "data.fasta")

    train_fasta = os.path.join(PROJECT_ROOT, "data", CELL_NAME, "train", "data.fasta")
    test_fasta  = os.path.join(PROJECT_ROOT, "data", CELL_NAME, "test", "data.fasta")

    feat_train_dir = os.path.join(PROJECT_ROOT, "feature", CELL_NAME, "train")
    feat_test_dir  = os.path.join(PROJECT_ROOT, "feature", CELL_NAME, "test")

    if not os.path.exists(input_fasta):
        raise FileNotFoundError(f"找不到输入 fasta：{input_fasta}")

    # 1) split 7:1
    records = read_fasta_records(input_fasta)
    if len(records) < 2:
        raise RuntimeError(f"FASTA 记录数太少：{len(records)}")

    random.seed(SEED)
    random.shuffle(records)
    n = len(records)
    n_test = max(1, int(round(n * TEST_FRACTION)))
    test_records = records[:n_test]
    train_records = records[n_test:]

    print(f"[Split] total={n}, train={len(train_records)}, test={len(test_records)} (ratio~7:1)")
    write_fasta_records(train_fasta, train_records)
    write_fasta_records(test_fasta, test_records)

    # 2) iLearn 脚本路径
    python_exe = sys.executable
    ilearn_basic = os.path.join(ILEARN_DIR, "iLearn-nucleotide-basic.py")
    ilearn_pse   = os.path.join(ILEARN_DIR, "iLearn-nucleotide-Pse.py")
    if not os.path.exists(ilearn_basic):
        raise FileNotFoundError(f"找不到：{ilearn_basic}")
    if not os.path.exists(ilearn_pse):
        raise FileNotFoundError(f"找不到：{ilearn_pse}")

    # 3) 生成 CKSNAP / RCKmer / (TNC->tpcp)
    def gen_basic(method, out_name):
        raw_train = os.path.join(feat_train_dir, out_name.replace(".csv", "_raw.csv"))
        raw_test  = os.path.join(feat_test_dir,  out_name.replace(".csv", "_raw.csv"))
        out_train = os.path.join(feat_train_dir, out_name)
        out_test  = os.path.join(feat_test_dir,  out_name)

        run_ilearn(python_exe, ilearn_basic, train_fasta, method, raw_train)
        run_ilearn(python_exe, ilearn_basic, test_fasta,  method, raw_test)

        convert_raw_keep_header_drop_namecol(raw_train, out_train)
        convert_raw_keep_header_drop_namecol(raw_test,  out_test)

    # CKSNAP
    gen_basic("CKSNAP", "cksnap.csv")
    # RCKmer
    gen_basic("RCKmer", "rckmer.csv")

    # TPCP（这里用 TNC 代替生成到 tpcp.csv）
    if USE_TNC_AS_TPCP:
        gen_basic("TNC", "tpcp.csv")
        print("[WARN] 当前用 TNC 代替 TPCP 生成 tpcp.csv（文件名不变，但含义不同）")

    # 4) PseTNC：用 iLearn-nucleotide-Pse.py（SCPseTNC）
    def gen_pse(method, out_name):
        raw_train = os.path.join(feat_train_dir, out_name.replace(".csv", "_raw.csv"))
        raw_test  = os.path.join(feat_test_dir,  out_name.replace(".csv", "_raw.csv"))
        out_train = os.path.join(feat_train_dir, out_name)
        out_test  = os.path.join(feat_test_dir,  out_name)

        # iLearn Pse 需要 --type DNA
        run_ilearn(python_exe, ilearn_pse, train_fasta, method, raw_train, extra_args=["--type", "DNA"])
        run_ilearn(python_exe, ilearn_pse, test_fasta,  method, raw_test,  extra_args=["--type", "DNA"])

        convert_raw_keep_header_drop_namecol(raw_train, out_train)
        convert_raw_keep_header_drop_namecol(raw_test,  out_test)

    gen_pse(PSETNC_METHOD, "psetnc.csv")

    # 5) mismatch k-mer（纯 Python）
    generate_mismatch_profile_csv(train_fasta, os.path.join(feat_train_dir, "mismatch.csv"), k=MISMATCH_K, m=MISMATCH_M)
    generate_mismatch_profile_csv(test_fasta,  os.path.join(feat_test_dir,  "mismatch.csv"), k=MISMATCH_K, m=MISMATCH_M)

    print("\n[DONE] 已生成特征到：")
    print("  " + feat_train_dir)
    print("  " + feat_test_dir)


if __name__ == "__main__":
    main()
