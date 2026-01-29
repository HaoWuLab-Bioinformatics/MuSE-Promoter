# -*- coding: utf-8 -*-
"""
【直接复制粘贴可用】数据集构建 + 特征生成（修复版）
修复点：
1) ✅ Word2Vec 只用训练集序列训练（杜绝 test 泄露）
2) ✅ 使用 pos/neg 两个 .fa 文件，强制标签（pos=1, neg=0），不依赖 header
3) ✅ iLearn CSV 清洗按列名删除 label/class/y（防止标签列混入特征）
4) ✅ 行数对齐：以 y.txt 行数为准，对齐所有特征文件（避免 cat 报错）
注意：只修改数据输入，生成文件与保存位置保持不变（feature/E.coli/train|test/...）
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

# 项目根目录（你的目录结构：脚本旁边有 iPro-WAEL-main）
BASE_PROJECT_DIR = os.path.join(SCRIPT_DIR, "iPro-WAEL-main")

DATA_ROOT_DIR = os.path.join(BASE_PROJECT_DIR, "data")
FEATURE_ROOT_DIR = os.path.join(BASE_PROJECT_DIR, "feature")

# iLearn 脚本名（自动寻路用）
ILEARN_SCRIPT_NAME = "iLearn-nucleotide-basic.py"
# ====== 数据集配置 ======
DATASETS_CONFIG = {
    "R.capsulatus": {
        "fasta": "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/data/R.capsulatus/data.fasta",
        "label": "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/data/R.capsulatus/y.txt",
    }
}


# ================= 4) 数据准备（修复科学计数法问题）=================
def prepare_dataset(name, config):
    print(f"\n--- 准备数据集: {name} ---")

    fasta_path = config["fasta"]
    label_path = config["label"]

    if not os.path.exists(fasta_path) or not os.path.exists(label_path):
        print(f"[ERROR] 找不到输入文件: {fasta_path} 或 {label_path}")
        return [], [], []

    # 1. 读取序列
    headers, seqs = [], []
    current_seq = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if current_seq:
                    seqs.append("".join(current_seq))
                headers.append(line[1:])
                current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            seqs.append("".join(current_seq))

    # 2. 读取标签 (修复点：处理科学计数法 1.0000e+00)
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    # 先转 float 再转 int，可以兼容 "1", "1.0", "1.000e+00"
                    val = int(float(line))
                    labels.append(val)
                except ValueError:
                    print(f"    [Warning] 跳过非法标签行: {line}")

    # 3. 校验对齐
    if len(seqs) != len(labels):
        print(f"[ERROR] 行数不匹配！Fasta序列: {len(seqs)} 条, 标签: {len(labels)} 行")
        min_len = min(len(seqs), len(labels))
        headers = headers[:min_len]
        seqs = seqs[:min_len]
        labels = labels[:min_len]

    pos_cnt = sum(1 for y in labels if y == 1)
    neg_cnt = sum(1 for y in labels if y == 0)
    print(f"    [Stat] 原始样本 - 正: {pos_cnt}, 负: {neg_cnt}")

    # 4. 类别平衡 (下采样)
    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    neg_idx = [i for i, y in enumerate(labels) if y == 0]
    target = min(len(pos_idx), len(neg_idx))

    random.seed(SPLIT_SEED)
    if len(pos_idx) > target:
        pos_idx = random.sample(pos_idx, target)
    if len(neg_idx) > target:
        neg_idx = random.sample(neg_idx, target)

    pick = pos_idx + neg_idx
    random.shuffle(pick)

    out_h = [headers[i] for i in pick]
    out_s = [seqs[i] for i in pick]
    out_l = [labels[i] for i in pick]
    return out_h, out_s, out_l



# 划分参数
SPLIT_SEED = 10
TRAIN_TO_TEST = (7, 1)

# mismatch
DNA_ALPHABET = ["A", "C", "G", "T"]

# Word2Vec
W2V_KMER = 5
W2V_VECTOR_DIM = 100
W2V_OUT_LEN = 8000


# ================= 小工具：行数对齐 =================
def _count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", errors="ignore") as f:
        return sum(1 for _ in f)


def _pad_or_trim_txt(path, target_lines):
    """
    将按行存储的特征 txt 对齐到 target_lines：
    - 行数多了：截断
    - 行数少了：补全0行（补的维度=该文件已存在行的最大列数；如果文件为空就补单个0）
    """
    if not os.path.exists(path):
        return False

    with open(path, "r", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    max_cols = 1
    for ln in lines:
        if ln.strip():
            cols = len(ln.split())
            if cols > max_cols:
                max_cols = cols

    if len(lines) > target_lines:
        lines = lines[:target_lines]
    elif len(lines) < target_lines:
        pad_line = " ".join(["0"] * max_cols)
        lines.extend([pad_line] * (target_lines - len(lines)))

    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    return True


def align_feature_rows(out_dir):
    """
    以 y.txt 为基准，将 out_dir 下所有特征文件行数对齐。
    """
    y_path = os.path.join(out_dir, "y.txt")
    if not os.path.exists(y_path):
        print(f"[Align] 缺少 y.txt，跳过对齐：{out_dir}")
        return

    target = _count_lines(y_path)
    if target <= 0:
        print(f"[Align] y.txt 行数为0，跳过对齐：{out_dir}")
        return

    feats = ["cksnap.txt", "mismatch.txt", "rckmer.txt", "psetnc.txt", "tpcp.txt", "word2vec.txt"]
    print(f"[Align] 以 y.txt={target} 行为基准，对齐特征文件：{out_dir}")

    for fn in feats:
        fp = os.path.join(out_dir, fn)
        if os.path.exists(fp):
            before = _count_lines(fp)
            _pad_or_trim_txt(fp, target)
            after = _count_lines(fp)
            if before != after:
                print(f"    [Align] {fn}: {before} -> {after}")
        else:
            print(f"    [Align-Warn] 缺少 {fn}，跳过")


# ================= 1) label 解析（保留；pos/neg 强制标签不靠它）=================
def parse_label_from_header(header: str):
    """
    尝试从 fasta header 解析二分类标签。
    返回 0/1 或 None（解析不出来）
    """
    h = (header or "").strip()
    hl = h.lower()

    m = re.search(r"\|([01])(\||$)", h)
    if m:
        return int(m.group(1))

    m = re.search(r"\b(label|class|y)\b\s*[:= ]\s*([01])\b", hl)
    if m:
        return int(m.group(2))

    if any(k in hl for k in ["non_prom", "nonprom", "nonpromoter", "negative", "_neg"]):
        return 0
    if any(k in hl for k in ["promoter", "positive", "_pos"]):
        return 1

    return None


def read_fasta_raw_strict(
    fasta_path,
    default_label=None,
    allow_missing_labels=False,
    missing_label_fill=None,
):
    """
    读取 fasta，允许 A/C/G/T/N（大小写）。

    - default_label != None：强制全文件统一标签
    - default_label == None：从 header 解析标签（本脚本的 E.coli 用不到）
    """
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"[FASTA Missing] {fasta_path}")

    headers, seqs, labels = [], [], []
    header = None
    seq_parts = []
    VALID_CHARS = set("ACGTNacgtn")

    def finalize_one(h, parts):
        if h is None:
            return
        full_seq = "".join(parts)
        if len(full_seq) == 0:
            return
        if not all(c in VALID_CHARS for c in full_seq):
            return

        headers.append(h)
        seqs.append(full_seq)

        if default_label is not None:
            labels.append(int(default_label))
        else:
            labels.append(parse_label_from_header(h))  # 可能 None

    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finalize_one(header, seq_parts)
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
    finalize_one(header, seq_parts)

    if default_label is None:
        total = len(labels)
        if total == 0:
            raise ValueError(f"[FASTA Empty] {fasta_path} 没有可用序列")

        missing_cnt = sum(1 for y in labels if y is None)
        print(f"    [Debug-read] header无法解析标签(None)数量: {missing_cnt}/{total} ({(missing_cnt/total):.2%})")

        if missing_cnt > 0:
            if not allow_missing_labels:
                preview = []
                for h, y in zip(headers, labels):
                    if y is None:
                        preview.append(h)
                        if len(preview) >= 5:
                            break
                raise ValueError(
                    f"[LabelMissing] {fasta_path} 有 {missing_cnt}/{total} 条 header 无法解析出 0/1 标签。\n"
                    f"示例(前5条):\n" + "\n".join([f"  >{x}" for x in preview]) + "\n"
                    f"请确保 header 包含 |0 或 |1，或 label=0/1、class=0/1、y=0/1 等。"
                )
            if missing_label_fill is None:
                missing_label_fill = 1
            labels = [int(missing_label_fill) if y is None else int(y) for y in labels]
        else:
            labels = [int(y) for y in labels]

    return headers, seqs, labels


# ================= 2) 写 fasta + y（保存位置不变）=================
def write_fasta_and_y(headers, seqs, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fasta_path = os.path.join(out_dir, "data.fasta")  # ✅ 输出仍叫 data.fasta（按你原流程不变）
    y_path = os.path.join(out_dir, "y.txt")
    with open(fasta_path, "w", encoding="utf-8") as wf, open(y_path, "w", encoding="utf-8") as wy:
        for h, s, l in zip(headers, seqs, labels):
            clean_h = (h.split()[0]).replace("|", "_")
            wf.write(f">{clean_h}|{int(l)}|training\n{s}\n")
            wy.write(f"{int(l)}\n")
    return fasta_path


# ================= 3) 分层划分 =================
def stratified_split_data(headers, seqs, labels, seed=SPLIT_SEED, ratio=TRAIN_TO_TEST):
    data = list(zip(headers, seqs, labels))
    random.seed(seed)
    random.shuffle(data)

    pos_data = [d for d in data if d[2] == 1]
    neg_data = [d for d in data if d[2] == 0]

    if len(pos_data) == 0 or len(neg_data) == 0:
        print("[ERROR] 数据集中缺少某一类样本，无法分层划分！")
        return ([], [], []), ([], [], [])

    train_r, test_r = ratio
    total = train_r + test_r

    n_pos_test = int(len(pos_data) * (test_r / total))
    n_neg_test = int(len(neg_data) * (test_r / total))

    if len(pos_data) >= 2 and n_pos_test == 0:
        n_pos_test = 1
    if len(neg_data) >= 2 and n_neg_test == 0:
        n_neg_test = 1

    if n_pos_test >= len(pos_data):
        n_pos_test = len(pos_data) - 1
    if n_neg_test >= len(neg_data):
        n_neg_test = len(neg_data) - 1

    test_set = pos_data[:n_pos_test] + neg_data[:n_neg_test]
    train_set = pos_data[n_pos_test:] + neg_data[n_neg_test:]

    random.shuffle(train_set)
    random.shuffle(test_set)

    def unzip(ds):
        if not ds:
            return [], [], []
        h, s, l = zip(*ds)
        return list(h), list(s), list(l)

    return unzip(train_set), unzip(test_set)



# ================= 5) iLearn 寻路 + 运行 =================
def find_ilearn_paths(start_dir, script_name):
    print(f"[寻路] 正在自动搜索 {script_name} ...")
    start_path = Path(start_dir)
    search_bases = [start_path] + list(start_path.parents)[:3]

    found_basic = None
    found_pse = None
    for base in search_bases:
        try:
            results = list(base.rglob(script_name))
            if results:
                found_basic = str(results[0])
                ilearn_dir = os.path.dirname(found_basic)
                pse_script = os.path.join(ilearn_dir, "iLearn-nucleotide-Pse.py")
                if os.path.exists(pse_script):
                    found_pse = pse_script
                break
        except Exception:
            continue

    if found_basic and found_pse:
        print(f"[成功] 找到 iLearn:\n    Basic: {found_basic}\n    Pse:   {found_pse}")
        return found_basic, found_pse
    return None, None


def run_cmd(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"[ERROR] iLearn 运行出错:\n{r.stderr}")
        return False
    return True


def sanitize_ilearn_csv_safe(in_csv, out_csv):
    if not os.path.exists(in_csv):
        return False

    with open(in_csv, "r", encoding="utf-8", errors="ignore") as fin:
        reader = csv.reader(fin)
        rows = list(reader)

    if not rows:
        return False

    header = rows[0]
    if len(header) <= 1:
        return False

    header_l = [h.strip().lower() for h in header]

    DROP_KEYS = {"label", "class", "y", "target", "output", "classlabel", "classname"}
    ID_KEYS = {"id", "name", "samplename", "seqname", "sequence", "sample", "index"}

    drop_idx = set()

    for i, col in enumerate(header_l):
        if col in DROP_KEYS:
            drop_idx.add(i)
        if any(col.startswith(k + "_") for k in DROP_KEYS):
            drop_idx.add(i)

    if header_l[0] in ID_KEYS or any(header_l[0].startswith(k) for k in ID_KEYS):
        drop_idx.add(0)

    keep_idx = [i for i in range(len(header)) if i not in drop_idx]
    if not keep_idx:
        raise ValueError(f"[iLearn Sanitize] {in_csv} 清洗后没有可用特征列，请检查表头：{header}")

    with open(out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow([header[i] for i in keep_idx])
        for row in rows[1:]:
            if len(row) < len(header):
                continue
            writer.writerow([row[i] for i in keep_idx])

    return True


def csv_to_pure_txt_keep_all(csv_path, txt_path):
    if not os.path.exists(csv_path):
        return False

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        return False

    header = rows[0]
    data_rows = rows[1:]
    if not data_rows:
        return False

    with open(txt_path, "w", encoding="utf-8") as f:
        for row in data_rows:
            if len(row) < len(header):
                continue
            out = []
            ok = True
            for x in row:
                x = x.strip()
                if x == "":
                    out.append("0")
                else:
                    try:
                        float(x)
                        out.append(x)
                    except ValueError:
                        ok = False
                        break
            if ok:
                f.write(" ".join(out) + "\n")
    return True


def generate_ilearn_features(data_dir, headers, seqs, labels, ilearn_basic, ilearn_pse):
    tmp_fasta = os.path.join(data_dir, "temp_ilearn.fasta")
    with open(tmp_fasta, "w", encoding="utf-8") as f:
        for i, (h, s, y) in enumerate(zip(headers, seqs, labels)):
            safe_h = h.replace("|", "_").replace(" ", "_").replace(",", "")
            f.write(f">{safe_h}_{i}|{int(y)}|training\n{s}\n")

    py = sys.executable
    tasks = [
        ("CKSNAP", ilearn_basic, "cksnap"),
        ("RCKmer", ilearn_basic, "rckmer"),
        ("TNC", ilearn_basic, "tpcp"),
        ("SCPseTNC", ilearn_pse, "psetnc"),
    ]

    for method, script, name in tasks:
        raw_csv = os.path.join(data_dir, f"_raw_{name}.csv")
        clean_csv = os.path.join(data_dir, f"{name}.csv")
        final_txt = os.path.join(data_dir, f"{name}.txt")

        if os.path.exists(final_txt):
            os.remove(final_txt)
        if os.path.exists(raw_csv):
            os.remove(raw_csv)
        if os.path.exists(clean_csv):
            os.remove(clean_csv)

        args = ["--file", tmp_fasta, "--method", method, "--format", "csv", "--out", raw_csv]
        if method == "SCPseTNC":
            args += ["--type", "DNA"]

        ok = run_cmd([py, script] + args)
        if not ok or not os.path.exists(raw_csv):
            print(f"    [FAIL] {name}: iLearn 未生成 CSV")
            continue

        sanitize_ilearn_csv_safe(raw_csv, clean_csv)

        if csv_to_pure_txt_keep_all(clean_csv, final_txt):
            print(f"    [OK] 生成: {name}.txt")
        else:
            print(f"    [FAIL] {name}: CSV 转 TXT 失败")

        try:
            os.remove(raw_csv)
        except Exception:
            pass
        try:
            os.remove(clean_csv)
        except Exception:
            pass

    try:
        os.remove(tmp_fasta)
    except Exception:
        pass


# ================= 6) mismatch 特征 =================
def build_kmers(k):
    return ["".join(p) for p in product(DNA_ALPHABET, repeat=k)]


def get_mismatch_neighbors(s, m):
    if m == 0:
        return [s]
    res = {s}
    for i in range(len(s)):
        for b in DNA_ALPHABET:
            if b != s[i]:
                res.add(s[:i] + b + s[i + 1:])
    return list(res)


def generate_mismatch_txt(seqs, out_txt, k=5, m=1):
    kmers = build_kmers(k)
    kmer_idx = {km: i for i, km in enumerate(kmers)}
    num_kmers = len(kmers)

    with open(out_txt, "w", encoding="utf-8") as f:
        for seq in seqs:
            seq = (seq or "").upper()
            vec = [0.0] * num_kmers
            n = len(seq) - k + 1
            if n > 0:
                for i in range(n):
                    sub = seq[i:i + k]
                    if any(c not in DNA_ALPHABET for c in sub):
                        continue
                    for neighbor in get_mismatch_neighbors(sub, m):
                        j = kmer_idx.get(neighbor, None)
                        if j is not None:
                            vec[j] += 1.0
                vec = [v / n for v in vec]
            f.write(" ".join(f"{v:.6f}" for v in vec) + "\n")


# ================= 7) Word2Vec：只用训练集训练 =================
def train_and_save_word2vec_from_seqs(seqs, index_path, w2v_path, k=W2V_KMER, dim=W2V_VECTOR_DIM):
    print(f"[W2V] 正在基于训练集 {len(seqs)} 条序列训练 Word2Vec ...")
    try:
        from gensim.models import Word2Vec
    except ImportError:
        raise ImportError("缺少 gensim 库，请运行: pip install gensim")

    sentences = []
    for seq in seqs:
        seq = (seq or "").upper()
        if len(seq) < k:
            continue
        kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        if kmers:
            sentences.append(kmers)

    if not sentences:
        raise ValueError("[W2V] 没有有效的 k-mer 序列用于训练（训练集可能太小或序列太短）")

    model = Word2Vec(
        sentences,
        vector_size=dim,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
    )

    vocab_keys = list(model.wv.index_to_key)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(w2v_path), exist_ok=True)

    with open(index_path, "w", encoding="utf-8") as f_idx:
        f_idx.write(" ".join(vocab_keys))

    with open(w2v_path, "w", encoding="utf-8") as f_vec:
        for key in vocab_keys:
            vec = model.wv[key]
            f_vec.write(" ".join([f"{v:.6f}" for v in vec]) + "\n")

    print(f"[W2V] 保存完成：\n    index: {index_path}\n    vecs : {w2v_path}")
    return True


def load_w2v_resources(index_path, w2v_path):
    if not os.path.exists(index_path) or not os.path.exists(w2v_path):
        return None, None

    print(f"[W2V] 加载资源：{os.path.basename(index_path)}, {os.path.basename(w2v_path)}")
    with open(index_path, "r", encoding="utf-8") as f:
        idx_list = f.read().strip().split()

    rows = []
    with open(w2v_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split()])

    if not rows:
        return None, None
    return idx_list, torch.tensor(rows, dtype=torch.float32)


def generate_w2v_txt(seqs, idx_list, w2v_mat, out_path, k=W2V_KMER, out_len=W2V_OUT_LEN):
    kmer_map = {kmer: i for i, kmer in enumerate(idx_list)}
    pool = nn.AdaptiveAvgPool1d(out_len)

    feats = []
    for seq in seqs:
        seq = (seq or "").upper()
        if len(seq) < k:
            ids = [0]
        else:
            ids = [kmer_map.get(seq[i:i + k], 0) for i in range(len(seq) - k + 1)]
            if not ids:
                ids = [0]

        emb = w2v_mat[torch.tensor(ids, dtype=torch.long)]
        flat = emb.reshape(-1).view(1, 1, -1)

        if flat.size(-1) == 0:
            out = torch.zeros(out_len)
        else:
            out = pool(flat).view(-1)
        feats.append(out)

    if not feats:
        return False

    mat = torch.stack(feats, dim=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in mat:
            f.write(" ".join(f"{x:.6g}" for x in row.tolist()) + "\n")
    return True


# ================= 主程序 =================
def main():
    ilearn_basic, ilearn_pse = find_ilearn_paths(BASE_PROJECT_DIR, ILEARN_SCRIPT_NAME)
    if not ilearn_basic:
        print("\n[致命错误] 无法找到 iLearn 脚本！")
        return

    for ds_name, config in DATASETS_CONFIG.items():
        print(f"\n########################################")
        print(f"### 处理数据集: {ds_name}")
        print(f"########################################")

        all_h, all_s, all_l = prepare_dataset(ds_name, config)
        if not all_h:
            continue

        (tr_h, tr_s, tr_l), (te_h, te_s, te_l) = stratified_split_data(all_h, all_s, all_l)
        print(f"    [Split] Train集: {len(tr_l)}, Test集: {len(te_l)}")

        ds_out_root = os.path.join(FEATURE_ROOT_DIR, ds_name)

        w2v_index = os.path.join(ds_out_root, "w2v_index.txt")
        w2v_vecs = os.path.join(ds_out_root, "w2v_vecs.txt")

        try:
            train_and_save_word2vec_from_seqs(tr_s, w2v_index, w2v_vecs)
            idx_list, w2v_mat = load_w2v_resources(w2v_index, w2v_vecs)
        except Exception as e:
            print(f"[W2V] 训练/加载失败，将跳过 W2V 特征生成。原因：{e}")
            idx_list, w2v_mat = None, None

        split_dirs = {
            "train": (os.path.join(ds_out_root, "train"), tr_h, tr_s, tr_l),
            "test": (os.path.join(ds_out_root, "test"), te_h, te_s, te_l),
        }

        for split_type, (out_dir, h, s, l) in split_dirs.items():
            h, s, l = list(h), list(s), list(l)
            if len(l) == 0:
                print(f"    [Skip] {ds_name} 的 {split_type} 集为空，跳过特征生成。")
                continue

            print(f"\n    >>> 生成 {split_type} 特征 -> {out_dir}")

            write_fasta_and_y(h, s, l, out_dir)

            generate_ilearn_features(out_dir, h, s, l, ilearn_basic, ilearn_pse)

            generate_mismatch_txt(s, os.path.join(out_dir, "mismatch.txt"))
            print(f"    [OK] 生成: mismatch.txt")

            if idx_list is not None and w2v_mat is not None:
                ok = generate_w2v_txt(s, idx_list, w2v_mat, os.path.join(out_dir, "word2vec.txt"))
                if ok:
                    print(f"    [OK] 生成: word2vec.txt")
                else:
                    print(f"    [FAIL] 生成 word2vec.txt 失败")
            else:
                print(f"    [Skip] 缺少 Word2Vec 资源，跳过 word2vec.txt")

            align_feature_rows(out_dir)

    print("\n=== 全部任务完成 ===")


if __name__ == "__main__":
    main()
