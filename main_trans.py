# main_trans.py
import os
import csv
import torch

import Weighted_average_trans
import RF
import trans_pytorch as CNN   # ✅ 修复：Transformer 分支用 CNN.pred 调用


# =========================
# Path helpers
# =========================
ROOT = os.path.dirname(os.path.abspath(__file__))

def pjoin(*parts):
    return os.path.join(ROOT, *parts)

def find_file_case_insensitive(path: str):
    """
    若 path 不存在，尝试在同目录下按大小写不敏感匹配同名文件。
    返回找到的实际路径，否则返回 None。
    """
    if os.path.exists(path):
        return path
    d = os.path.dirname(path)
    base = os.path.basename(path)
    if not os.path.isdir(d):
        return None
    base_lower = base.lower()
    for fn in os.listdir(d):
        if fn.lower() == base_lower:
            cand = os.path.join(d, fn)
            if os.path.isfile(cand):
                return cand
    return None

def must_exist(path: str, hint: str = ""):
    real = find_file_case_insensitive(path)
    if real is None:
        msg = f"[FileMissing] 找不到文件：{path}"
        if hint:
            msg += f"\n提示：{hint}"
        raise FileNotFoundError(msg)
    return real


# =========================
# Loaders
# =========================
def load_txt_vector(path):
    path = must_exist(path, "请确认 data/<cell_line>/{train,test}/y_*.txt 是否存在")
    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals.append(float(line))
    return torch.tensor(vals, dtype=torch.float32)

def load_csv_matrix_skip_first_col(path):
    """
    读取 CSV，跳过第1列（通常是样本ID），其余列转 float。
    自动跳过空行/表头（非数字行）。
    """
    path = must_exist(
        path,
        "你需要先生成 iLearnPlus 特征：cksnap/mismatch/rckmer/psetnc/tpcp，并放到 feature/<cell_line>/{train,test}/ 下"
    )

    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # 跳过全空
            if all((x is None or str(x).strip() == "") for x in row):
                continue
            # 跳过第1列（样本id）
            feats = row[1:]
            # 有些 CSV 可能带表头，float 转换会失败，这里直接跳过
            try:
                rows.append([float(x) for x in feats])
            except ValueError:
                # 表头/非数字行，跳过
                continue

    if len(rows) == 0:
        raise RuntimeError(f"[BadCSV] 读取到 0 行有效数字特征：{path}\n"
                           f"请检查该文件是否为空、是否有表头、是否确实包含数值列。")

    return torch.tensor(rows, dtype=torch.float32)

def load_whitespace_matrix(path):
    """
    读取空格/制表分隔矩阵（用于 word2vec.txt）
    """
    path = must_exist(
        path,
        "请先运行 feature_code.py 生成 word2vec 特征，或确认 feature/<cell_line>/{train,test}/word2vec.txt 已存在"
    )
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                # 非数字行直接跳过
                continue

    if len(rows) == 0:
        raise RuntimeError(f"[BadTXT] 读取到 0 行有效 word2vec 特征：{path}\n"
                           f"请检查该文件是否为空、是否为纯数字矩阵。")

    return torch.tensor(rows, dtype=torch.float32)


# =========================
# Utils
# =========================
def normalization_minmax(data):
    minv = data.min(dim=0).values
    maxv = data.max(dim=0).values
    rng = maxv - minv
    rng[rng == 0] = 1.0
    return (data - minv) / rng

def nor_train_test(x_train, x_test):
    x = torch.cat([x_train, x_test], dim=0)
    x = normalization_minmax(x)
    return x[:x_train.shape[0]], x[x_train.shape[0]:]

def stratified_split_multi(X1, X2, y, test_size=1/8, seed=10):
    """
    分层划分：返回
      X1_train, X1_weight, X2_train, X2_weight, y_train, y_weight
    """
    g = torch.Generator().manual_seed(seed)
    y_int = y.to(torch.int64).view(-1)

    idx0 = torch.nonzero(y_int == 0, as_tuple=False).view(-1)
    idx1 = torch.nonzero(y_int == 1, as_tuple=False).view(-1)

    idx0 = idx0[torch.randperm(idx0.numel(), generator=g)] if idx0.numel() > 0 else idx0
    idx1 = idx1[torch.randperm(idx1.numel(), generator=g)] if idx1.numel() > 0 else idx1

    n0_w = max(1, int(idx0.numel() * test_size)) if idx0.numel() > 0 else 0
    n1_w = max(1, int(idx1.numel() * test_size)) if idx1.numel() > 0 else 0

    w_idx = torch.cat([idx0[:n0_w], idx1[:n1_w]], dim=0) if (n0_w + n1_w) > 0 else torch.tensor([], dtype=torch.long)
    tr_idx = torch.cat([idx0[n0_w:], idx1[n1_w:]], dim=0)

    if w_idx.numel() == 0:
        raise RuntimeError("[SplitError] weight set 为空：可能某个类别样本太少。请调小 test_size 或检查 y_train 标签。")

    w_idx = w_idx[torch.randperm(w_idx.numel(), generator=g)]
    tr_idx = tr_idx[torch.randperm(tr_idx.numel(), generator=g)]

    return X1[tr_idx], X1[w_idx], X2[tr_idx], X2[w_idx], y[tr_idx], y[w_idx]

import torch

def _to_tensor(x, dtype=None):
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype)
    return t

def accuracy(y_true, y_pred):
    y_true = _to_tensor(y_true, torch.int64).view(-1)
    y_pred = _to_tensor(y_pred, torch.int64).view(-1)
    return (y_true == y_pred).float().mean().item()

def mcc(y_true, y_pred):
    y_true = _to_tensor(y_true, torch.int64).view(-1)
    y_pred = _to_tensor(y_pred, torch.int64).view(-1)

    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / (denom ** 0.5)

def roc_auc_score_torch(y_true, y_score):
    # ✅ 关键：不管传进来是什么，都转成 torch tensor
    y_true = _to_tensor(y_true, torch.int64).view(-1)
    y_score = _to_tensor(y_score, torch.float32).view(-1)

    order = torch.argsort(y_score, descending=True)
    y_true = y_true[order]

    P = (y_true == 1).sum().item()
    N = (y_true == 0).sum().item()
    if P == 0 or N == 0:
        return 0.0

    tps = torch.cumsum((y_true == 1).to(torch.float32), dim=0)
    fps = torch.cumsum((y_true == 0).to(torch.float32), dim=0)
    tpr = tps / P
    fpr = fps / N

    tpr = torch.cat([torch.tensor([0.0]), tpr])
    fpr = torch.cat([torch.tensor([0.0]), fpr])

    return torch.trapz(tpr, fpr).item()

def EvaluateMetrics(y_test, label, proba, tag=""):
    # ✅ 同样先统一类型，避免 rf/trans/fusion 输出类型不一致
    y_test_t = _to_tensor(y_test, torch.int64).view(-1)
    label_t  = _to_tensor(label, torch.int64).view(-1)
    proba_t  = _to_tensor(proba, torch.float32).view(-1)

    aucv = roc_auc_score_torch(y_test_t, proba_t)
    accv = accuracy(y_test_t, label_t)
    mccv = mcc(y_test_t, label_t)

    prefix = f"[{tag}] " if tag else ""
    print(prefix + "AUC=%.6f ACC=%.6f MCC=%.6f" % (aucv, accv, mccv))

# =========================
# Main
# =========================
def main():
    # 改这里切换 cell line
    cell_lines = "GM12878"

    # --- labels ---
    y_train = load_txt_vector(pjoin("data", cell_lines, "train", "y_train.txt"))
    y_test  = load_txt_vector(pjoin("data", cell_lines, "test",  "y_test.txt"))

    # --- feature list (RF) ---
    features = ["cksnap", "mismatch", "rckmer", "psetnc", "tpcp"]

    # --- load RF train features ---
    x_train_rf = None
    for feature in features:
        fp = pjoin("feature", cell_lines, "train", f"{feature}.csv")
        fea = load_csv_matrix_skip_first_col(fp)
        x_train_rf = fea if x_train_rf is None else torch.cat([x_train_rf, fea], dim=1)

    # --- load RF test features ---
    x_test_rf = None
    for feature in features:
        fp = pjoin("feature", cell_lines, "test", f"{feature}.csv")
        fea = load_csv_matrix_skip_first_col(fp)
        x_test_rf = fea if x_test_rf is None else torch.cat([x_test_rf, fea], dim=1)

    # --- load word2vec (Transformer branch) ---
    x_train_cnn = load_whitespace_matrix(pjoin("feature", cell_lines, "train", "word2vec.txt"))
    x_test_cnn  = load_whitespace_matrix(pjoin("feature", cell_lines, "test",  "word2vec.txt"))

    # --- normalize RF ---
    x_train_rf, x_test_rf = nor_train_test(x_train_rf, x_test_rf)

    # --- split weight set ---
    x_train_rf, x_weight_rf, x_train_cnn, x_weight_cnn, y_train2, y_weight = stratified_split_multi(
        x_train_rf, x_train_cnn, y_train, test_size=1/8, seed=10
    )

    print("x_train_rf:", tuple(x_train_rf.shape))
    print("x_test_rf :", tuple(x_test_rf.shape))
    print("x_train_tr:", tuple(x_train_cnn.shape))
    print("x_test_tr :", tuple(x_test_cnn.shape))

    # --- params ---
    cnn_lr = 0.001
    cnn_KERNEL_SIZE = 11
    cnn_KERNEL_NUM = 32
    n_trees = 300

    # --- RF ---
    rf_weight_proba, rf_test_proba, rf_test_class = RF.pred(
        x_train_rf, y_train2, x_weight_rf, x_test_rf, n_trees
    )

    # --- Transformer (CNN placeholder) ---
    cnn_weight_proba, cnn_test_proba, cnn_test_class = CNN.pred(
        x_train_cnn, y_train2, x_weight_cnn, x_test_cnn,
        cnn_lr, cnn_KERNEL_NUM, cnn_KERNEL_SIZE
    )

    EvaluateMetrics(y_test, rf_test_class, rf_test_proba, tag="RF")
    EvaluateMetrics(y_test, cnn_test_class, cnn_test_proba, tag="Transformer")

    # --- Weighted average fusion ---
    proba, label = Weighted_average_trans.weight(
        y_weight, rf_weight_proba, cnn_weight_proba, rf_test_proba, cnn_test_proba
    )
    EvaluateMetrics(y_test, label, proba, tag="Fusion")


if __name__ == "__main__":
    main()
