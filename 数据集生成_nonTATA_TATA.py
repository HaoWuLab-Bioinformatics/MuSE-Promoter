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
# 1. 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 【修改点】项目根目录在里面那一层
BASE_PROJECT_DIR = os.path.join(SCRIPT_DIR, "iPro-WAEL-main")

# 2. 目录设置
DATA_ROOT_DIR = os.path.join(BASE_PROJECT_DIR, "data")
FEATURE_ROOT_DIR = os.path.join(BASE_PROJECT_DIR, "feature")

# 3. Word2Vec 资源路径
INDEX_FILE = os.path.join(BASE_PROJECT_DIR, "index_promoters.txt")
W2V_FILE = os.path.join(BASE_PROJECT_DIR, "word2vec_promoters.txt")

# 4. iLearn 脚本
ILEARN_SCRIPT_NAME = "iLearn-nucleotide-basic.py"

# 5. 数据集配置
DATASETS_CONFIG = {
    "mouse TATA": {
        "pos": os.path.join(DATA_ROOT_DIR, "mouse TATA", "Mouse_tata.fasta"),
        "neg": os.path.join(DATA_ROOT_DIR, "mouse TATA", "Mouse_nonprom.fasta")
    },
    "mouse nonTATA": {
        # single 文件：header 中应包含 |1 或 |0
        "single": os.path.join(DATA_ROOT_DIR, "mouse nonTATA", "data.fasta")
    }
}

# 6. 参数
SPLIT_SEED = 10
TRAIN_TO_TEST = (7, 1)
DNA_ALPHABET = ["A", "C", "G", "T"]
W2V_KMER = 5
W2V_VECTOR_DIM = 100

def _count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", errors="ignore") as f:
        return sum(1 for _ in f)

def _pad_or_trim_txt(path, target_lines):
    """
    将一个按行存储的特征 txt 对齐到 target_lines：
    - 行数多了：截断
    - 行数少了：补全0行（补的维度=该文件已存在行的最大列数；如果文件为空就补单个0）
    """
    if not os.path.exists(path):
        return False

    # 读全部行
    with open(path, "r", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # 估计列数（用于补零行）
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

    with open(path, "w") as f:
        for ln in lines:
            f.write(ln + "\n")
    return True

def align_feature_rows(out_dir):
    """
    以 y.txt 为基准，将 out_dir 下所有特征文件行数对齐。
    解决下游 torch.cat 653/654 问题。
    """
    y_path = os.path.join(out_dir, "y.txt")
    if not os.path.exists(y_path):
        print(f"[Align] 缺少 y.txt，跳过对齐：{out_dir}")
        return

    target = _count_lines(y_path)
    if target <= 0:
        print(f"[Align] y.txt 行数为0，跳过对齐：{out_dir}")
        return

    # 你这套流程里会出现的特征文件
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

# ================= 1. Word2Vec 自动训练集成模块 =================
def train_and_save_word2vec(seqs, index_path, w2v_path, k=W2V_KMER, dim=W2V_VECTOR_DIM):
    """如果缺少资源，使用 gensim 现场训练"""
    print(f"[W2V] 正在基于 {len(seqs)} 条序列训练 Word2Vec 模型...")
    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("[ERROR] 缺少 gensim 库，请运行: pip install gensim")
        return False

    sentences = []
    for seq in seqs:
        seq = seq.upper()
        kmer_list = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        if kmer_list:
            sentences.append(kmer_list)

    if not sentences:
        print("[ERROR] 没有有效的 k-mer 序列用于训练。")
        return False

    model = Word2Vec(sentences, vector_size=dim, window=5, min_count=1, workers=4, sg=1)

    vocab_keys = list(model.wv.index_to_key)

    print(f"[W2V] 保存索引到: {os.path.basename(index_path)}")
    with open(index_path, "w") as f_idx:
        f_idx.write(" ".join(vocab_keys))

    print(f"[W2V] 保存向量到: {os.path.basename(w2v_path)}")
    with open(w2v_path, "w") as f_vec:
        for key in vocab_keys:
            vec = model.wv[key]
            line = " ".join([f"{v:.6f}" for v in vec])
            f_vec.write(line + "\n")

    print("[W2V] 训练完成！")
    return True


def ensure_word2vec_resources(all_configs):
    """检查资源，如果不存在则收集所有序列进行训练"""
    if os.path.exists(INDEX_FILE) and os.path.exists(W2V_FILE):
        return True

    print("[检测] 未找到 Word2Vec 预训练文件，准备开始生成...")
    all_seqs = []

    for name, cfg in all_configs.items():
        paths = []
        if "pos" in cfg:
            paths.append(cfg["pos"])
        if "neg" in cfg:
            paths.append(cfg["neg"])
        if "single" in cfg:
            paths.append(cfg["single"])

        for p in paths:
            if os.path.exists(p):
                with open(p, 'r', errors='ignore') as f:
                    lines = f.readlines()
                seq = ""
                for line in lines:
                    line = line.strip()
                    if line.startswith(">"):
                        if seq:
                            all_seqs.append(seq)
                        seq = ""
                    else:
                        seq += line
                if seq:
                    all_seqs.append(seq)

    if not all_seqs:
        print("[ERROR] 找不到任何序列文件，无法训练 W2V。")
        return False

    return train_and_save_word2vec(all_seqs, INDEX_FILE, W2V_FILE)


# ================= 2. 核心工具函数 =================
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
    else:
        return None, None
def read_fasta_raw(
    fasta_path,
    default_label=None,
    infer_half_if_missing=True,
    missing_ratio_threshold=0.9
):
    """
    读取 fasta，允许 A/C/G/T/N（大小写）。

    - default_label != None：强制全文件统一标签
    - default_label == None：尝试从 header 解析标签，解析不到先记为 None
      若解析不到的比例 >= missing_ratio_threshold 且 infer_half_if_missing=True，
      则启用“前半 1 后半 0”的自动标签规则（覆盖所有标签）。
    """
    if not os.path.exists(fasta_path):
        print(f"[WARNING] 文件未找到: {fasta_path}")
        return [], [], []

    headers, seqs, labels = [], [], []
    header = None
    seq_parts = []

    # ✅ 允许 N
    VALID_CHARS = set("ACGTNacgtn")

    def finalize_one(h, parts):
        if h is None:
            return
        full_seq = "".join(parts)
        if len(full_seq) == 0:
            return
        # 只要属于 A/C/G/T/N 就保留
        if not all(c in VALID_CHARS for c in full_seq):
            return

        headers.append(h)
        seqs.append(full_seq)

        if default_label is not None:
            labels.append(int(default_label))
        else:
            y = parse_label_from_header(h)
            labels.append(y)  # 可能是 0/1/None

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

    # ====== 自动补标签：half rule（关键修复）======
    if default_label is None and infer_half_if_missing and len(labels) > 0:
        missing_cnt = sum(1 for y in labels if y is None)
        ratio = missing_cnt / len(labels)
        print(f"    [Debug-read] header无法解析标签(None)数量: {missing_cnt}/{len(labels)} ({ratio:.2%})")

        if ratio >= missing_ratio_threshold:
            mid = len(labels) // 2
            labels = [1] * mid + [0] * (len(labels) - mid)
            print(f"    [Debug-read] 触发 half-label：前 {mid} 条=1，后 {len(labels)-mid} 条=0")

    # 如果还有 None（说明没触发 half-label 且解析不全），这里兜底为 1（也可改成丢弃）
    if default_label is None:
        none_left = sum(1 for y in labels if y is None)
        if none_left > 0:
            print(f"    [Debug-read] 仍有 {none_left} 条标签为 None，已兜底为 1")
            labels = [1 if y is None else int(y) for y in labels]
        else:
            labels = [int(y) for y in labels]

    return headers, seqs, labels



def write_fasta_and_y(headers, seqs, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fasta_path = os.path.join(out_dir, "data.fasta")
    y_path = os.path.join(out_dir, "y.txt")
    with open(fasta_path, "w", encoding="utf-8") as wf, open(y_path, "w", encoding="utf-8") as wy:
        for h, s, l in zip(headers, seqs, labels):
            clean_h = h.split()[0].replace("|", "_")
            wf.write(f">{clean_h}|{int(l)}|training\n{s}\n")
            wy.write(f"{int(l)}\n")
    return fasta_path

def stratified_split_data(headers, seqs, labels, seed=SPLIT_SEED, ratio=TRAIN_TO_TEST):
    """
    分层划分，尽量保证 test 集里正负样本都至少有 1 条（当样本量允许时）
    """
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

    # 先按比例算
    n_pos_test = int(len(pos_data) * (test_r / total))
    n_neg_test = int(len(neg_data) * (test_r / total))

    # ✅ 保底：如果某类>=2，但算出来为0，则至少给1个到test
    if len(pos_data) >= 2 and n_pos_test == 0:
        n_pos_test = 1
    if len(neg_data) >= 2 and n_neg_test == 0:
        n_neg_test = 1

    # ✅ 再保底：test不能把某类拿光，至少给train留1个
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



# ================= iLearn & Mismatch =================
def run_cmd(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"[ERROR] iLearn 运行出错: {r.stderr}")
        return False
    return True


def sanitize_ilearn_csv(in_csv, out_csv):
    if not os.path.exists(in_csv):
        return False
    with open(in_csv, "r") as fin, open(out_csv, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        rows = list(reader)
        if not rows:
            return False
        header = rows[0]
        if len(header) > 1:
            writer.writerow(header[1:])
        for row in rows[1:]:
            if len(row) > 1:
                writer.writerow(row[1:])
    return True


def csv_to_pure_txt(csv_path, txt_path):
    if not os.path.exists(csv_path):
        return False
    rows_to_write = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        all_rows = list(reader)
        if len(all_rows) < 2:
            return False
        for row in all_rows[1:]:
            if len(row) > 1:
                rows_to_write.append(row[1:])
    if not rows_to_write:
        return False
    with open(txt_path, 'w', encoding='utf-8') as f:
        for row in rows_to_write:
            f.write(" ".join(row) + "\n")
    return True


def generate_ilearn_features(data_dir, headers, seqs, labels, ilearn_basic, ilearn_pse):
    tmp_fasta = os.path.join(data_dir, "temp_ilearn.fasta")
    with open(tmp_fasta, "w") as f:
        for i, (h, s, y) in enumerate(zip(headers, seqs, labels)):
            safe_h = h.replace("|", "_").replace(" ", "_").replace(",", "")
            f.write(f">{safe_h}_{i}|{int(y)}|training\n{s}\n")

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
        if os.path.exists(final_txt):
            os.remove(final_txt)

        args = ["--file", tmp_fasta, "--method", method, "--format", "csv", "--out", raw_csv]
        if method == "SCPseTNC":
            args += ["--type", "DNA"]

        if run_cmd([py, script] + args):
            if os.path.exists(raw_csv):
                sanitize_ilearn_csv(raw_csv, clean_csv)
                if csv_to_pure_txt(clean_csv, final_txt):
                    print(f"    [OK] 生成: {name}.txt")
                os.remove(raw_csv)
                if os.path.exists(clean_csv):
                    os.remove(clean_csv)
            else:
                print(f"    [FAIL] 未生成 {name} CSV 文件")
        else:
            print(f"    [FAIL] 运行 {method} 失败")

    if os.path.exists(tmp_fasta):
        os.remove(tmp_fasta)


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
    with open(out_txt, "w") as f:
        for seq in seqs:
            seq = seq.upper()
            vec = [0.0] * num_kmers
            n = len(seq) - k + 1
            if n > 0:
                for i in range(n):
                    sub = seq[i:i + k]
                    # mismatch 特征这里仍然只考虑 A/C/G/T（含N的 kmer 直接跳过）
                    if any(c not in DNA_ALPHABET for c in sub):
                        continue
                    for neighbor in get_mismatch_neighbors(sub, m):
                        if neighbor in kmer_idx:
                            vec[kmer_idx[neighbor]] += 1.0
                vec = [v / n for v in vec]
            f.write(" ".join(f"{v:.6f}" for v in vec) + "\n")


# ================= Word2Vec 加载/生成 =================
def load_w2v_resources():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(W2V_FILE):
        return None, None
    print(f"[资源] 加载 Word2Vec 矩阵...")
    with open(INDEX_FILE, "r") as f:
        idx_list = f.read().strip().split()
    rows = []
    with open(W2V_FILE, "r") as f:
        for line in f:
            if line.strip():
                rows.append([float(x) for x in line.split()])
    return idx_list, torch.tensor(rows, dtype=torch.float32)


def generate_w2v_txt(seqs, idx_list, w2v_mat, out_path, k=W2V_KMER, out_len=8000):
    kmer_map = {kmer: i for i, kmer in enumerate(idx_list)}
    pool = nn.AdaptiveAvgPool1d(out_len)
    feats = []
    for seq in seqs:
        seq = seq.upper()
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

    if feats:
        mat = torch.stack(feats, dim=0)
        with open(out_path, "w") as f:
            for row in mat:
                f.write(" ".join(f"{x:.6g}" for x in row.tolist()) + "\n")
        return True
    return False


# ================= ✅ 准备数据逻辑（支持 single + 快速验证）=================
def prepare_dataset(name, config):
    print(f"\n--- 准备数据集: {name} ---")

    headers, seqs, labels = [], [], []

    # 情况1：pos/neg 分开
    if "pos" in config and "neg" in config:
        print(f"    [Load] 正样本: {os.path.basename(config['pos'])}")
        ph, ps, pl = read_fasta_raw(config['pos'], default_label=1)

        print(f"    [Load] 负样本: {os.path.basename(config['neg'])}")
        nh, ns, nl = read_fasta_raw(config['neg'], default_label=0)

        headers = ph + nh
        seqs = ps + ns
        labels = pl + nl

    # 情况2：single 文件（自己带标签）
    elif "single" in config:
        print(f"    [Load] 单文件: {os.path.basename(config['single'])}")
        headers, seqs, labels = read_fasta_raw(config["single"], default_label=None)

    # ✅ 快速验证（临时调试）
    pos_cnt = sum(1 for y in labels if y == 1)
    neg_cnt = sum(1 for y in labels if y == 0)
    print(f"    [Debug] labels[:20] = {labels[:20]}")
    print(f"    [Debug] pos_cnt={pos_cnt}, neg_cnt={neg_cnt}")
    if "single" in config:
        print("    [Debug] headers[:10] 预览：")
        for h in headers[:10]:
            print(f"        >{h}")

    print(f"    [Stat] 正样本数: {pos_cnt}, 负样本数: {neg_cnt}")
    if pos_cnt == 0 or neg_cnt == 0:
        print(f"[ERROR] {name} 数据缺失（可能原因：标签未被解析成0/1，或序列读取为空）。")
        return [], [], []

    # 平衡
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
import re

def parse_label_from_header(header: str):
    """
    尝试从 fasta header 解析二分类标签。
    返回 0/1 或 None（解析不出来）
    """
    h = (header or "").strip()
    hl = h.lower()

    # 常见：|0 或 |1（例如 xxx|1|training）
    m = re.search(r"\|([01])(\||$)", h)
    if m:
        return int(m.group(1))

    # 形如 label:0 / label=1 / class 0 / y=1
    m = re.search(r"\b(label|class|y)\b\s*[:= ]\s*([01])\b", hl)
    if m:
        return int(m.group(2))

    # 关键词（可按你的数据扩展）
    if any(k in hl for k in ["non_prom", "nonprom", "nonpromoter", "negative", "_neg"]):
        return 0
    if any(k in hl for k in ["promoter", "positive", "_pos"]):
        return 1

    return None


# ================= 主程序 =================
def main():
    ilearn_basic, ilearn_pse = find_ilearn_paths(BASE_PROJECT_DIR, ILEARN_SCRIPT_NAME)
    if not ilearn_basic:
        print("\n[致命错误] 无法找到 iLearn 脚本！")
        return

    ensure_word2vec_resources(DATASETS_CONFIG)

    idx_list, w2v_mat = load_w2v_resources()
    if idx_list is None:
        print("[Warning] Word2Vec 加载失败，将跳过 W2V 特征生成。")

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
        split_dirs = {
            "train": (os.path.join(ds_out_root, "train"), tr_h, tr_s, tr_l),
            "test": (os.path.join(ds_out_root, "test"), te_h, te_s, te_l)
        }

        for split_type, (out_dir, h, s, l) in split_dirs.items():
            # 这里 h/s/l 本来就是 list（我上面 stratified_split_data 已返回 list），保险起见再转一次
            h, s, l = list(h), list(s), list(l)

            if len(l) == 0:
                print(f"    [Skip] {ds_name} 的 {split_type} 集为空，跳过特征生成。")
                continue

            print(f"\n    >>> 生成 {split_type} 特征 -> {out_dir}")
            write_fasta_and_y(h, s, l, out_dir)

            # iLearn 特征
            generate_ilearn_features(out_dir, h, s, l, ilearn_basic, ilearn_pse)

            # mismatch
            generate_mismatch_txt(s, os.path.join(out_dir, "mismatch.txt"))
            print(f"    [OK] 生成: mismatch.txt")

            # word2vec
            if idx_list is not None:
                if generate_w2v_txt(s, idx_list, w2v_mat, os.path.join(out_dir, "word2vec.txt")):
                    print(f"    [OK] 生成: word2vec.txt")
            else:
                print(f"    [Skip] 缺少 Word2Vec 资源，跳过。")
            align_feature_rows(out_dir)

    print("\n=== 全部任务完成 ===")
def assert_single_has_labels(dataset_name, labels, fallback_cnt, threshold=0.9):
    """
    当 single 文件大部分样本都解析不到标签时，直接停止该数据集处理，避免生成垃圾数据。
    """
    total = len(labels)
    if total == 0:
        print(f"[ERROR] {dataset_name} single 文件读取为空。")
        return False
    ratio = fallback_cnt / total
    if ratio >= threshold:
        print(f"[ERROR] {dataset_name} single 文件几乎没有可解析标签：fallback={fallback_cnt}/{total} ({ratio:.2%})")
        print("        说明该 data.fasta 很可能不是带0/1标签的分类数据，需要另提供标签文件或重新构建 nonTATA 数据。")
        return False
    return True


if __name__ == "__main__":
    main()
