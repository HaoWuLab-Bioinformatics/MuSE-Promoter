import os
import csv
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score
import pandas as pd

# ==========================================
# 假设这些模块你本地有，保持引用
try:
    import Weighted_average_trans
    import RF
except ImportError:
    print("[Warning] 缺少 Weighted_average_trans 或 RF 模块，部分功能可能无法运行。")
# ==========================================

ROOT = os.path.dirname(os.path.abspath(__file__))
TARGET_DATASETS = ["mouse TATA", "mouse nonTATA"]


def pjoin(*parts):
    return os.path.join(ROOT, *parts)


def set_seed(seed: int = 2025):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def must_exist(path: str, hint: str = ""):
    if os.path.exists(path): return path
    d = os.path.dirname(path)
    base = os.path.basename(path)
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.lower() == base.lower(): return os.path.join(d, fn)
    msg = f"[FileMissing] 找不到文件：{path}"
    if hint: msg += f"\n提示：{hint}"
    raise FileNotFoundError(msg)


# ==========================================
# 模型架构 (保持不变)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 15000):
        super().__init__()
        self.d_model = d_model
        pe = self._build_pe(max_len, d_model)
        self.register_buffer("pe", pe)

    def _build_pe(self, length, d_model):
        pe = torch.zeros(1, length, d_model)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        L = x.size(1)
        if L > self.pe.size(1): self.pe = self._build_pe(L, self.d_model).to(x.device)
        return x + self.pe[:, :L, :]


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=11, padding=5)
        self.conv4 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.conv1(x);
        c2 = self.conv2(x);
        c3 = self.conv3(x);
        c4 = self.conv4(x)
        out = torch.cat([c1, c2, c3, c4], dim=1)
        return self.relu(self.bn(out))


class HybridTransformer(nn.Module):
    def __init__(self, seq_len=8000, feat_dim=1258, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(1, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32))
        self.cnn_encoder = nn.Sequential(MultiScaleCNN(32, 64), nn.MaxPool1d(4), SEBlock(64),
                                         nn.Conv1d(64, d_model, kernel_size=5, stride=1, padding=2),
                                         nn.BatchNorm1d(d_model), nn.ReLU(), nn.MaxPool1d(4), SEBlock(d_model))
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                                                   dropout=dropout, activation="gelu", batch_first=True,
                                                   norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.feat_mlp = nn.Sequential(nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(512, 128), nn.ReLU())
        fusion_dim = d_model + 128
        self.classifier = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Dropout(dropout), nn.Linear(fusion_dim, 64),
                                        nn.GELU(), nn.Linear(64, 1))

    def forward(self, x_seq, x_feat):
        x = x_seq.unsqueeze(-1)
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        x = self.cnn_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        seq_emb = x.mean(dim=1)
        feat_emb = self.feat_mlp(x_feat)
        combined = torch.cat([seq_emb, feat_emb], dim=1)
        return self.classifier(combined)


def run_hybrid_training(X_seq_tr, X_feat_tr, y_tr, X_seq_w, X_feat_w, y_w, X_seq_te, X_feat_te):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [Hybrid] Device: {device} | Seq Len: {X_seq_tr.shape[1]} | Feat Dim: {X_feat_tr.shape[1]}")

    # 1. 转换数据 (Train)
    X_seq_tr = torch.as_tensor(X_seq_tr, dtype=torch.float32).to(device)
    X_feat_tr = torch.as_tensor(X_feat_tr, dtype=torch.float32).to(device)
    y_tr = torch.as_tensor(y_tr, dtype=torch.float32).view(-1, 1).to(device)

    # 2. 转换数据 (Validation - 使用传入的 Weight Set)
    X_seq_val = torch.as_tensor(X_seq_w, dtype=torch.float32).to(device)
    X_feat_val = torch.as_tensor(X_feat_w, dtype=torch.float32).to(device)
    # 【关键修改】这里使用 y_w
    y_val = torch.as_tensor(y_w, dtype=torch.float32).view(-1, 1).to(device)

    # 3. 构建 Dataset (不再进行内部切分，最大化利用训练数据)
    train_ds = TensorDataset(X_seq_tr, X_feat_tr, y_tr)
    val_ds = TensorDataset(X_seq_val, X_feat_val, y_val)

    # 4. DataLoader (保留 drop_last=True 修复 Batch=1 问题)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    # 注意：如果 drop_last 导致小样本集(如45个)变成0个batch，需要特殊处理
    if len(train_ds) < 64:
        print("  [Warn] Sample size < Batch size. Forcing batch_size to len(train_ds)")
        train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True, drop_last=False)

    val_loader = DataLoader(val_ds, batch_size=128)

    # 5. 模型初始化
    model = HybridTransformer(
        seq_len=X_seq_tr.shape[1],
        feat_dim=X_feat_tr.shape[1],
        d_model=128, nhead=4, num_layers=2, dropout=0.3
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    best_loss = float('inf')
    best_state = None
    patience = 10
    bad = 0

    # 6. 训练循环
    for epoch in range(100):
        model.train()
        for b_seq, b_feat, b_y in train_loader:
            opt.zero_grad()
            logits = model(b_seq, b_feat)
            loss = loss_fn(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        # 如果 val_loader 为空（极小概率），防报错
        if len(val_loader) > 0:
            with torch.no_grad():
                for b_seq, b_feat, b_y in val_loader:
                    logits = model(b_seq, b_feat)
                    val_loss += loss_fn(logits, b_y).item()
            avg_val_loss = val_loss / len(val_loader)
        else:
            avg_val_loss = 0.0

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state: model.load_state_dict(best_state)
    model.eval()

    # 7. 预测
    def predict(x_s, x_f):
        ds = TensorDataset(torch.as_tensor(x_s, dtype=torch.float32), torch.as_tensor(x_f, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=128, shuffle=False)
        probs = []
        with torch.no_grad():
            for b_s, b_f in dl:
                b_s, b_f = b_s.to(device), b_f.to(device)
                p = torch.sigmoid(model(b_s, b_f))
                probs.append(p.cpu())
        # 防止空数据报错
        if len(probs) == 0: return torch.zeros((0, 1))
        return torch.cat(probs, dim=0).view(-1, 1)

    weight_proba = predict(X_seq_w, X_feat_w)
    test_proba = predict(X_seq_te, X_feat_te)
    test_class = (test_proba >= 0.5).long()

    return weight_proba, test_proba, test_class


# ==========================================
# 工具函数
# ==========================================
def load_txt_vector(path):
    path = must_exist(path)
    vals = []
    with open(path, "r") as f:
        for line in f:
            if line.strip(): vals.append(float(line.strip()))
    return torch.tensor(vals, dtype=torch.float32)


def load_csv_matrix_skip_first_col(path):
    # 读取CSV并转为Matrix
    path = must_exist(path)
    try:
        # 尝试 index_col=0 模式
        df = pd.read_csv(path, header=0, index_col=0)
    except:
        # 回退模式
        df = pd.read_csv(path, header=None)
        if isinstance(df.iloc[0, 0], str): df = df.iloc[1:, :]
        df = df.iloc[:, 1:]

    # 强转数值
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return torch.tensor(df.values, dtype=torch.float32)


def load_whitespace_matrix(path):
    # 读取空格分隔的TXT并转为Matrix
    path = must_exist(path)
    raw_data = []
    max_len = 0
    with open(path, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                parts = [float(x) for x in line.split()]
                if len(parts) > 0:
                    raw_data.append(parts)
                    if len(parts) > max_len: max_len = len(parts)
            except ValueError:
                continue
    if not raw_data: raise ValueError(f"File empty: {path}")
    # 自动 padding
    for i in range(len(raw_data)):
        curr = len(raw_data[i])
        if curr < max_len:
            raw_data[i].extend([0.0] * (max_len - curr))
    return torch.tensor(raw_data, dtype=torch.float32)


def normalization_minmax(data):
    minv = data.min(dim=0).values;
    maxv = data.max(dim=0).values
    rng = maxv - minv;
    rng[rng == 0] = 1.0
    return (data - minv) / rng


def stratified_split_multi(X1, X2, y, test_size=1 / 8, seed=10):
    g = torch.Generator().manual_seed(seed)
    y_int = y.long().view(-1)
    idx0 = torch.nonzero(y_int == 0, as_tuple=False).view(-1)
    idx1 = torch.nonzero(y_int == 1, as_tuple=False).view(-1)
    idx0 = idx0[torch.randperm(idx0.numel(), generator=g)]
    idx1 = idx1[torch.randperm(idx1.numel(), generator=g)]
    n0 = int(len(idx0) * test_size);
    n1 = int(len(idx1) * test_size)
    w_idx = torch.cat([idx0[:n0], idx1[:n1]])
    tr_idx = torch.cat([idx0[n0:], idx1[n1:]])
    w_idx = w_idx[torch.randperm(len(w_idx), generator=g)]
    tr_idx = tr_idx[torch.randperm(len(tr_idx), generator=g)]
    return X1[tr_idx], X1[w_idx], X2[tr_idx], X2[w_idx], y[tr_idx], y[w_idx]


def EvaluateMetrics(y_test, y_pred_class, y_pred_prob, tag=""):
    # 定义一个内部函数，自动识别并转换为 numpy 数组
    def _to_numpy(x):
        if hasattr(x, 'detach'): # 如果是 PyTorch Tensor
            return x.detach().cpu().numpy().flatten()
        elif isinstance(x, np.ndarray): # 如果已经是 NumPy Array
            return x.flatten()
        else: # 或者是 List 等其他类型
            return np.array(x).flatten()

    # 统一转换
    y_true = _to_numpy(y_test)
    y_cls = _to_numpy(y_pred_class)
    y_prob = _to_numpy(y_pred_prob)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
    acc = accuracy_score(y_true, y_cls)
    mcc = matthews_corrcoef(y_true, y_cls)
    print(f"  [{tag}] AUC={auc:.4f} ACC={acc:.4f} MCC={mcc:.4f}")
    return auc, acc, mcc
# ==========================================
# 关键修复：安全对齐函数
# ==========================================
def safe_align_tensors(tensor_list, label_tensor=None):
    """
    找到所有 tensor 中的最小行数，并将所有 tensor 截断到该行数。
    解决 'Sizes of tensors must match except in dimension 1' 错误。
    """
    # 1. 获取每个 tensor 的行数 (dim 0)
    sizes = [t.shape[0] for t in tensor_list]

    if label_tensor is not None:
        sizes.append(label_tensor.shape[0])

    min_len = min(sizes)

    # 2. 检查是否有不匹配
    if any(s != min_len for s in sizes):
        print(f"[Warning] Data size mismatch detected! Sizes: {sizes}")
        print(f"[Fix] Truncating all data to minimum length: {min_len}")

    # 3. 截断
    aligned_list = [t[:min_len] for t in tensor_list]

    if label_tensor is not None:
        aligned_label = label_tensor[:min_len]
        return aligned_list, aligned_label

    return aligned_list


# ==========================================
# Pipeline
# ==========================================
def run_pipeline_for_dataset(cell_name):
    print(f"\n==================================================")
    print(f" STARTING PIPELINE FOR: {cell_name}")
    print(f"==================================================")

    # 1. 加载标签
    try:
        y_train = load_txt_vector(pjoin("feature", cell_name, "train", "y.txt"))
        y_test = load_txt_vector(pjoin("feature", cell_name, "test", "y.txt"))
    except FileNotFoundError as e:
        print(f"[Error] 标签文件缺失: {e}")
        return

    # 2. 加载 Handcrafted Features (CSV/TXT)
    features = ["cksnap", "mismatch", "rckmer", "psetnc", "tpcp"]
    x_train_rf_list = []
    x_test_rf_list = []

    try:
        for feature in features:
            # Train set
            tr_csv = pjoin("feature", cell_name, "train", f"{feature}.csv")
            if os.path.exists(tr_csv):
                f_tr = load_csv_matrix_skip_first_col(tr_csv)
            else:
                tr_txt = pjoin("feature", cell_name, "train", f"{feature}.txt")
                f_tr = load_whitespace_matrix(tr_txt)
            x_train_rf_list.append(f_tr)

            # Test set
            te_csv = pjoin("feature", cell_name, "test", f"{feature}.csv")
            if os.path.exists(te_csv):
                f_te = load_csv_matrix_skip_first_col(te_csv)
            else:
                te_txt = pjoin("feature", cell_name, "test", f"{feature}.txt")
                f_te = load_whitespace_matrix(te_txt)
            x_test_rf_list.append(f_te)

    except Exception as e:
        print(f"[Error] 特征加载失败: {e}")
        return

    # 3. 加载 Word2Vec
    try:
        x_train_w2v = load_whitespace_matrix(pjoin("feature", cell_name, "train", "word2vec.txt"))
        x_test_w2v = load_whitespace_matrix(pjoin("feature", cell_name, "test", "word2vec.txt"))
    except Exception as e:
        print(f"[Error] Word2Vec 加载失败: {e}")
        return

    # 4. 加载 DNABERT (Optional)
    has_bert = False
    x_train_bert = None
    x_test_bert = None
    bert_path = pjoin("feature", cell_name, "train", "dnabert_features.txt")
    if os.path.exists(bert_path):
        try:
            x_train_bert = load_whitespace_matrix(bert_path)
            x_test_bert = load_whitespace_matrix(pjoin("feature", cell_name, "test", "dnabert_features.txt"))
            has_bert = True
        except:
            print("[Info] Failed loading BERT, skipping.")

    # =========================================================
    # 核心修复点：在 cat 之前，对所有特征进行对齐 (Alignment)
    # =========================================================

    # 收集训练集所有组件
    train_components = x_train_rf_list + [x_train_w2v]
    if has_bert: train_components.append(x_train_bert)

    # 执行对齐 (Training Set)
    print("Aligning Training Data...")
    aligned_train_comps, y_train = safe_align_tensors(train_components, y_train)

    # 还原回去
    num_rf = len(features)
    x_train_rf_list = aligned_train_comps[:num_rf]  # 前5个是RF
    x_train_w2v = aligned_train_comps[num_rf]  # 第6个是w2v
    if has_bert: x_train_bert = aligned_train_comps[num_rf + 1]

    # 收集测试集所有组件
    test_components = x_test_rf_list + [x_test_w2v]
    if has_bert: test_components.append(x_test_bert)

    # 执行对齐 (Test Set)
    print("Aligning Test Data...")
    aligned_test_comps, y_test = safe_align_tensors(test_components, y_test)

    # 还原回去
    x_test_rf_list = aligned_test_comps[:num_rf]
    x_test_w2v = aligned_test_comps[num_rf]
    if has_bert: x_test_bert = aligned_test_comps[num_rf + 1]

    # =========================================================
    # 现在可以安全地 cat 了
    # =========================================================

    x_train_rf = torch.cat(x_train_rf_list, dim=1)
    x_test_rf = torch.cat(x_test_rf_list, dim=1)

    # 归一化
    x_train_rf_norm = normalization_minmax(x_train_rf)
    min_v = x_train_rf.min(0).values;
    max_v = x_train_rf.max(0).values
    x_test_rf_norm = (x_test_rf - min_v) / (max_v - min_v + 1e-8)

    # 拼接 Sequence
    if has_bert:
        # BERT 维度对齐检查 (以防 w2v 和 bert 宽度不同，但这里主要是高度对齐)
        # 已经在 safe_align_tensors 处理了高度(行数)
        x_train_seq = torch.cat([x_train_w2v, x_train_bert], dim=1)
        x_test_seq = torch.cat([x_test_w2v, x_test_bert], dim=1)
    else:
        x_train_seq = x_train_w2v
        x_test_seq = x_test_w2v

    # 再次检查维度匹配 (Test集 sequence 的宽度必须和 Train 一样)
    if x_train_seq.shape[1] != x_test_seq.shape[1]:
        print(f"[Fix] Padding Test Sequence width: {x_test_seq.shape[1]} -> {x_train_seq.shape[1]}")
        tgt = x_train_seq.shape[1]
        if x_test_seq.shape[1] < tgt:
            pad = torch.zeros((x_test_seq.shape[0], tgt - x_test_seq.shape[1]))
            x_test_seq = torch.cat([x_test_seq, pad], dim=1)
        else:
            x_test_seq = x_test_seq[:, :tgt]

    print(f"[Data Ready] Train: {x_train_rf.shape[0]} samples. Test: {x_test_rf.shape[0]} samples.")

    # 6. 划分
    (x_rf_tr, x_rf_w, x_seq_tr, x_seq_w, y_tr, y_w) = stratified_split_multi(
        x_train_rf_norm, x_train_seq, y_train, test_size=1 / 8, seed=10
    )

    print("\n>>> Model 1: Random Forest")
    rf_weight_proba, rf_test_proba, rf_test_class = RF.pred(x_rf_tr, y_tr, x_rf_w, x_test_rf_norm, n_trees=300)
    EvaluateMetrics(y_test, rf_test_class, rf_test_proba, tag="RF")

    print("\n>>> Model 2: Hybrid Transformer")

    cnn_weight_proba, cnn_test_proba, cnn_test_class = run_hybrid_training(
        X_seq_tr=x_seq_tr, X_feat_tr=x_rf_tr, y_tr=y_tr,
        X_seq_w=x_seq_w, X_feat_w=x_rf_w, y_w=y_w,  # <--- 新增 y_w
        X_seq_te=x_test_seq, X_feat_te=x_test_rf_norm
    )
    EvaluateMetrics(y_test, cnn_test_class, cnn_test_proba, tag="Hybrid")

    print("\n>>> Model 3: Weighted Fusion")

    # --- 模型 3: Fusion ---
    print("\n>>> Model 3: Weighted Fusion")

    # 修改 to_cpu 函数，使其能处理 numpy array
    def to_tensor_cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        else:  # 如果是 numpy，转为 tensor
            return torch.tensor(x, dtype=torch.float32)

    proba, label = Weighted_average_trans.weight(
        to_tensor_cpu(y_w),
        to_tensor_cpu(rf_weight_proba), to_tensor_cpu(cnn_weight_proba),
        to_tensor_cpu(rf_test_proba), to_tensor_cpu(cnn_test_proba)
    )

    # EvaluateMetrics 已经修改为通用版，直接传即可
    res_auc, res_acc, res_mcc = EvaluateMetrics(y_test, label, proba, tag="Fusion")
    return res_auc, res_acc, res_mcc


def main():
    set_seed(2025)
    for dataset_name in TARGET_DATASETS:
        run_pipeline_for_dataset(dataset_name)


if __name__ == "__main__":
    main()