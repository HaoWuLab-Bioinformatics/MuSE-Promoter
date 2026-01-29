import os
import sys
import csv
import subprocess
import random
import torch
import torch.nn as nn
from itertools import product
from pathlib import Path

# ================= 配置区域 =================
# 1. 项目根目录
BASE_PROJECT_DIR = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main"
CELL_NAME = "K562"

# 2. 自动搜索 iLearn 的文件名 (不要改)
ILEARN_SCRIPT_NAME = "iLearn-nucleotide-basic.py"

# 3. 输入数据路径
DATA_DIR = os.path.join(BASE_PROJECT_DIR, "EPdata", CELL_NAME)
PROMOTER_FILE = os.path.join(DATA_DIR, "promoters.fasta")
ENHANCER_FILE = os.path.join(DATA_DIR, "enhancers.fasta")

# 4. 输出根目录
OUTPUT_ROOT = os.path.join(BASE_PROJECT_DIR, "EPfeature", CELL_NAME)

# 5. Word2Vec 资源路径
INDEX_FILE = os.path.join(BASE_PROJECT_DIR, "index_promoters.txt")
W2V_FILE = os.path.join(BASE_PROJECT_DIR, "word2vec_promoters.txt")

# 6. 参数设置
SPLIT_SEED = 10
TRAIN_TO_TEST = (7, 1)
DNA_ALPHABET = ["A", "C", "G", "T"]


# ================= 智能寻路函数 =================

def find_ilearn_paths(start_dir, script_name):
    """
    从 start_dir 开始，向上和向下搜索 iLearn 脚本
    """
    print(f"[寻路] 正在自动搜索 {script_name} ...")

    # 1. 搜索当前目录及子目录
    start_path = Path(start_dir)
    # 向上搜索 3 层
    search_bases = [start_path] + list(start_path.parents)[:3]

    found_basic = None
    found_pse = None

    for base in search_bases:
        # 在该目录下递归搜索 (限制深度防止太慢)
        try:
            # 搜索 iLearn-nucleotide-basic.py
            results = list(base.rglob(script_name))
            if results:
                found_basic = str(results[0])
                # 假设 Pse 脚本在同一个文件夹
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
    else:
        return None, None


# ================= 核心工具函数 =================

def read_fasta_raw(fasta_path):
    if not os.path.exists(fasta_path):
        print(f"[ERROR] 文件未找到: {fasta_path}")
        return [], []

    headers = []
    seqs = []
    header = None
    seq_parts = []

    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if header is not None:
                    headers.append(header)
                    seqs.append("".join(seq_parts))
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            headers.append(header)
            seqs.append("".join(seq_parts))
    return headers, seqs


def write_fasta_and_y(headers, seqs, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fasta_path = os.path.join(out_dir, "data.fasta")
    y_path = os.path.join(out_dir, "y.txt")

    with open(fasta_path, "w", encoding="utf-8") as wf, open(y_path, "w", encoding="utf-8") as wy:
        for h, s, l in zip(headers, seqs, labels):
            wf.write(f">{h}\n{s}\n")
            wy.write(f"{int(l)}\n")
    return fasta_path


def stratified_split_data(headers, seqs, labels, seed=SPLIT_SEED, ratio=TRAIN_TO_TEST):
    data = list(zip(headers, seqs, labels))
    random.seed(seed)
    random.shuffle(data)

    pos_data = [d for d in data if d[2] == 1]
    neg_data = [d for d in data if d[2] == 0]

    train_r, test_r = ratio
    total = train_r + test_r

    n_pos_test = int(len(pos_data) * (test_r / total))
    n_neg_test = int(len(neg_data) * (test_r / total))

    test_set = pos_data[:n_pos_test] + neg_data[:n_neg_test]
    train_set = pos_data[n_pos_test:] + neg_data[n_neg_test:]

    random.shuffle(train_set)
    random.shuffle(test_set)

    def unzip(ds):
        if not ds: return [], [], []
        return zip(*ds)

    return unzip(train_set), unzip(test_set)


# ================= 特征转换工具 =================

def csv_to_pure_txt(csv_path, txt_path):
    if not os.path.exists(csv_path): return False

    rows_to_write = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        all_rows = list(reader)
        if len(all_rows) < 2: return False

        for row in all_rows[1:]:
            if len(row) > 1:
                rows_to_write.append(row[1:])  # Skip label column

    if not rows_to_write: return False

    with open(txt_path, 'w', encoding='utf-8') as f:
        for row in rows_to_write:
            f.write(" ".join(row) + "\n")
    return True


# ================= Mismatch =================

def build_kmers(k):
    return ["".join(p) for p in product(DNA_ALPHABET, repeat=k)]


def get_mismatch_neighbors(s, m):
    if m == 0: return [s]
    res = {s}
    for i in range(len(s)):
        for b in DNA_ALPHABET:
            if b != s[i]: res.add(s[:i] + b + s[i + 1:])
    return list(res)


def generate_mismatch_txt(seqs, out_txt, k=5, m=1):
    kmers = build_kmers(k)
    kmer_idx = {km: i for i, km in enumerate(kmers)}
    num_kmers = len(kmers)

    with open(out_txt, "w") as f:
        for seq in seqs:
            seq = seq.upper()
            vec = [0.0] * num_kmers
            n = len(seq) - k + 1
            if n > 0:
                for i in range(n):
                    sub = seq[i:i + k]
                    if any(c not in DNA_ALPHABET for c in sub): continue
                    for neighbor in get_mismatch_neighbors(sub, m):
                        if neighbor in kmer_idx:
                            vec[kmer_idx[neighbor]] += 1.0
                vec = [v / n for v in vec]
            f.write(" ".join(f"{v:.6f}" for v in vec) + "\n")


# ================= iLearn 调用逻辑 =================

def run_cmd(cmd):
    # 打印简略命令
    print(f"    [EXEC] .../{os.path.basename(cmd[1])} {cmd[3]} ...")
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"[ERROR] iLearn 运行出错!")
        print(f"错误信息: {r.stderr}")
        return False
    return True


def sanitize_ilearn_csv(in_csv, out_csv):
    if not os.path.exists(in_csv): return False
    with open(in_csv, "r") as fin, open(out_csv, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        rows = list(reader)
        if not rows: return False

        header = rows[0]
        if len(header) > 1: writer.writerow(header[1:])

        for row in rows[1:]:
            if len(row) > 1: writer.writerow(row[1:])
    return True


def generate_ilearn_features(data_dir, headers, seqs, labels, ilearn_basic, ilearn_pse):
    tmp_fasta = os.path.join(data_dir, "temp_ilearn.fasta")
    with open(tmp_fasta, "w") as f:
        for h, s, y in zip(headers, seqs, labels):
            safe_h = h.replace("|", "_")
            f.write(f">{safe_h}|{int(y)}|training\n{s}\n")

    py = sys.executable
    tasks = [
        ("CKSNAP", ilearn_basic, "cksnap"),
        ("RCKmer", ilearn_basic, "rckmer"),
        ("TNC", ilearn_basic, "tpcp"),
        ("SCPseTNC", ilearn_pse, "psetnc")
    ]

    for method, script, name in tasks:
        raw_csv = os.path.join(data_dir, f"_raw_{name}.csv")
        clean_csv = os.path.join(data_dir, f"{name}.csv")
        final_txt = os.path.join(data_dir, f"{name}.txt")

        if os.path.exists(final_txt): os.remove(final_txt)

        args = ["--file", tmp_fasta, "--method", method, "--format", "csv", "--out", raw_csv]
        if method == "SCPseTNC": args += ["--type", "DNA"]

        if run_cmd([py, script] + args):
            if os.path.exists(raw_csv):
                sanitize_ilearn_csv(raw_csv, clean_csv)
                if csv_to_pure_txt(clean_csv, final_txt):
                    print(f"    [OK] 生成: {name}.txt")
                os.remove(raw_csv)
                if os.path.exists(clean_csv): os.remove(clean_csv)
            else:
                print(f"    [FAIL] 未生成 {name} CSV 文件")
        else:
            print(f"    [FAIL] 运行 {method} 失败")

    if os.path.exists(tmp_fasta): os.remove(tmp_fasta)


# ================= Word2Vec =================

def load_w2v_resources():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(W2V_FILE):
        return None, None
    print(f"[资源] 加载 Word2Vec 矩阵...")
    with open(INDEX_FILE, "r") as f:
        idx_list = f.read().strip().split()
    rows = []
    with open(W2V_FILE, "r") as f:
        for line in f:
            if line.strip(): rows.append([float(x) for x in line.split()])
    return idx_list, torch.tensor(rows, dtype=torch.float32)


def generate_w2v_txt(seqs, idx_list, w2v_mat, out_path, k=5, out_len=8000):
    kmer_map = {kmer: i for i, kmer in enumerate(idx_list)}
    pool = nn.AdaptiveAvgPool1d(out_len)

    feats = []
    for seq in seqs:
        ids = [kmer_map.get(seq[i:i + k], 0) for i in range(len(seq) - k + 1)]
        if not ids: ids = [0]
        emb = w2v_mat[torch.tensor(ids, dtype=torch.long)]
        flat = emb.reshape(-1).view(1, 1, -1)
        out = pool(flat).view(-1)
        feats.append(out)

    if feats:
        mat = torch.stack(feats, dim=0)
        with open(out_path, "w") as f:
            for row in mat:
                f.write(" ".join(f"{x:.6g}" for x in row.tolist()) + "\n")
        return True
    return False


# ================= 主程序 =================

def main():
    print(f"=== 开始处理 {CELL_NAME} ===")

    # --- 关键修改：自动寻找 iLearn ---
    ilearn_basic, ilearn_pse = find_ilearn_paths(BASE_PROJECT_DIR, ILEARN_SCRIPT_NAME)

    if not ilearn_basic:
        print("\n[致命错误] 无法找到 iLearn 脚本！")
        print(f"请确认您的项目中包含 '{ILEARN_SCRIPT_NAME}' 文件。")
        return

    # 1. 读取数据
    hp, sp = read_fasta_raw(PROMOTER_FILE)
    he, se = read_fasta_raw(ENHANCER_FILE)
    print(f"[数据] Promoters={len(sp)}, Enhancers={len(se)}")
    if not sp or not se: return

    # 2. 平衡
    target_num = min(len(sp), len(se))
    if len(se) > len(sp):
        print(f"[平衡] 下采样 Enhancers -> {len(sp)}")
        random.seed(SPLIT_SEED)
        idx = random.sample(range(len(se)), len(sp))
        he_bal, se_bal = [he[i] for i in idx], [se[i] for i in idx]
        hp_bal, sp_bal = hp, sp
    elif len(sp) > len(se):
        print(f"[平衡] 下采样 Promoters -> {len(se)}")
        random.seed(SPLIT_SEED)
        idx = random.sample(range(len(sp)), len(se))
        hp_bal, sp_bal = [hp[i] for i in idx], [sp[i] for i in idx]
        he_bal, se_bal = he, se
    else:
        hp_bal, sp_bal, he_bal, se_bal = hp, sp, he, se

    # 3. 合并 & 4. 划分
    all_h = hp_bal + he_bal
    all_s = sp_bal + se_bal
    all_l = [1] * len(sp_bal) + [0] * len(se_bal)

    (tr_h, tr_s, tr_l), (te_h, te_s, te_l) = stratified_split_data(all_h, all_s, all_l)
    print(f"[划分] Train={len(tr_l)}, Test={len(te_l)}")

    # 5. Word2Vec 资源
    idx_list, w2v_mat = load_w2v_resources()

    # 6. 生成所有文件
    dirs = {
        "train": (os.path.join(OUTPUT_ROOT, "train"), tr_h, tr_s, tr_l),
        "test": (os.path.join(OUTPUT_ROOT, "test"), te_h, te_s, te_l)
    }

    for split_name, (out_dir, h, s, l) in dirs.items():
        print(f"\n>>> 处理 {split_name} -> {out_dir}")
        write_fasta_and_y(h, s, l, out_dir)  # 1,2
        generate_ilearn_features(out_dir, h, s, l, ilearn_basic, ilearn_pse)  # 3,4,5,6
        generate_mismatch_txt(s, os.path.join(out_dir, "mismatch.txt"))  # 7
        print(f"    [OK] 生成: mismatch.txt")

        if idx_list is not None:
            if generate_w2v_txt(s, idx_list, w2v_mat, os.path.join(out_dir, "word2vec.txt")):  # 8
                print(f"    [OK] 生成: word2vec.txt")
        else:
            print(f"    [跳过] 缺少 word2vec 资源文件")

    print("\n=== 全部流程执行完毕 ===")


if __name__ == "__main__":
    main()