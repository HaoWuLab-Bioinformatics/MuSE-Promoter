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

# ==========================================
# 假设这些模块你本地有，保持引用
import Weighted_average_trans
import RF

# ==========================================

# 设置绝对路径根目录
PROJECT_ROOT = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/EPfeature"
CELL_NAME = "HeLa-S3"
DATA_ROOT = os.path.join(PROJECT_ROOT, CELL_NAME)


def set_seed(seed: int = 2025):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def must_exist(path: str):
    if os.path.exists(path):
        return path
    # 尝试模糊匹配文件名（不区分大小写）
    d = os.path.dirname(path)
    b = os.path.basename(path)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower() == b.lower():
                return os.path.join(d, f)
    raise FileNotFoundError(f"[FileMissing] 找不到文件：{path}")


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
        if L > self.pe.size(1):
            self.pe = self._build_pe(L, self.d_model).to(x.device)
        return x + self.pe[:, :L, :]


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

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
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(x)
        out = torch.cat([c1, c2, c3, c4], dim=1)
        return self.relu(self.bn(out))


class HybridTransformer(nn.Module):
    def __init__(self, seq_len=8000, feat_dim=1258, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        # Stream A
        self.input_proj = nn.Sequential(
            nn.Linear(1, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32)
        )
        self.cnn_encoder = nn.Sequential(
            MultiScaleCNN(32, 64), nn.MaxPool1d(4), SEBlock(64),
            nn.Conv1d(64, d_model, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.MaxPool1d(4), SEBlock(d_model)
        )
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Stream B
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU()
        )
        # Fusion
        fusion_dim = d_model + 128
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Dropout(dropout),
            nn.Linear(fusion_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )

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


def run_hybrid_training(X_seq_tr, X_feat_tr, y_tr, X_seq_w, X_feat_w, X_seq_te, X_feat_te):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hybrid] Training on {device}. Combined Sequence Dim: {X_seq_tr.shape[1]}")

    X_seq_tr = torch.as_tensor(X_seq_tr, dtype=torch.float32).to(device)
    X_feat_tr = torch.as_tensor(X_feat_tr, dtype=torch.float32).to(device)
    y_tr = torch.as_tensor(y_tr, dtype=torch.float32).view(-1, 1).to(device)

    X_seq_val = torch.as_tensor(X_seq_w, dtype=torch.float32).to(device)
    X_feat_val = torch.as_tensor(X_feat_w, dtype=torch.float32).to(device)

    n_total = X_seq_tr.size(0)
    n_val = int(n_total * 0.1)
    perm = torch.randperm(n_total)
    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    train_ds = TensorDataset(X_seq_tr[train_idx], X_feat_tr[train_idx], y_tr[train_idx])
    val_ds = TensorDataset(X_seq_tr[val_idx], X_feat_tr[val_idx], y_tr[val_idx])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

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
    patience = 12
    bad = 0

    for epoch in range(200):
        model.train()
        epoch_loss = 0
        for b_seq, b_feat, b_y in train_loader:
            opt.zero_grad()
            logits = model(b_seq, b_feat)
            loss = loss_fn(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_seq, b_feat, b_y in val_loader:
                logits = model(b_seq, b_feat)
                val_loss += loss_fn(logits, b_y).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    print(f"[Hybrid] Best Val Loss: {best_loss:.4f} (Stopped at epoch {epoch})")
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    def predict(x_s, x_f):
        ds = TensorDataset(torch.as_tensor(x_s, dtype=torch.float32), torch.as_tensor(x_f, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=128, shuffle=False)
        probs = []
        with torch.no_grad():
            for b_s, b_f in dl:
                b_s, b_f = b_s.to(device), b_f.to(device)
                p = torch.sigmoid(model(b_s, b_f))
                probs.append(p.cpu())
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


def load_whitespace_matrix(path):
    path = must_exist(path)
    rows = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                parts = [float(x) for x in line.split()]
                rows.append(parts)
            except ValueError:
                continue
    return torch.tensor(rows, dtype=torch.float32)


def normalization_minmax(data):
    minv = data.min(dim=0).values
    maxv = data.max(dim=0).values
    rng = maxv - minv
    rng[rng == 0] = 1.0
    return (data - minv) / rng


def align_and_truncate(data_dict):
    """
    自适应对齐所有张量的长度。
    输入 data_dict: {'name': tensor, 'name2': tensor ...}
    返回: 对齐后的张量列表
    """
    # 1. 获取所有张量的长度
    lengths = {k: v.shape[0] for k, v in data_dict.items() if v is not None}
    if not lengths:
        return {}

    # 2. 找到最小长度
    min_len = min(lengths.values())
    max_len = max(lengths.values())

    if min_len != max_len:
        print(f"\n[Warning] 检测到数据行数不一致! Min={min_len}, Max={max_len}")
        print(f"正在自动截断所有数据至 {min_len} 行以进行对齐...")
        for k, v in lengths.items():
            if v > min_len:
                print(f"  - {k}: {v} -> {min_len} (截断末尾 {v - min_len} 行)")

    # 3. 统一截断
    aligned_dict = {}
    for k, v in data_dict.items():
        if v is not None:
            aligned_dict[k] = v[:min_len]  # 截断多余行
        else:
            aligned_dict[k] = None

    return aligned_dict


def stratified_split_multi(X1, X2, y, test_size=1 / 8, seed=10):
    g = torch.Generator().manual_seed(seed)
    y_int = y.long().view(-1)
    idx0 = torch.nonzero(y_int == 0, as_tuple=False).view(-1)
    idx1 = torch.nonzero(y_int == 1, as_tuple=False).view(-1)
    idx0 = idx0[torch.randperm(idx0.numel(), generator=g)]
    idx1 = idx1[torch.randperm(idx1.numel(), generator=g)]
    n0 = int(len(idx0) * test_size)
    n1 = int(len(idx1) * test_size)
    w_idx = torch.cat([idx0[:n0], idx1[:n1]])
    tr_idx = torch.cat([idx0[n0:], idx1[n1:]])
    w_idx = w_idx[torch.randperm(len(w_idx), generator=g)]
    tr_idx = tr_idx[torch.randperm(len(tr_idx), generator=g)]
    return X1[tr_idx], X1[w_idx], X2[tr_idx], X2[w_idx], y[tr_idx], y[w_idx]


def EvaluateMetrics(y_test, y_pred_class, y_pred_prob, tag=""):
    def _to_numpy(x): return x.detach().cpu().view(-1).numpy() if isinstance(x, torch.Tensor) else np.array(x).flatten()

    y_true = _to_numpy(y_test)
    y_cls = _to_numpy(y_pred_class)
    y_prob = _to_numpy(y_pred_prob)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_cls)
    mcc = matthews_corrcoef(y_true, y_cls)
    print(f"[{tag}] AUC={auc:.6f} ACC={acc:.6f} MCC={mcc:.6f}")


# ==========================================
# Main
# ==========================================
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main():
    set_seed(2025)

    print(f"Processing Cell: {CELL_NAME}")
    print(f"Data Root: {DATA_ROOT}")

    train_dir = os.path.join(DATA_ROOT, "train")
    test_dir = os.path.join(DATA_ROOT, "test")

    # ================= 1. 加载所有数据 (Train) =================
    print("\n--- Loading Training Data ---")
    raw_train = {}

    # Labels
    raw_train['y'] = load_txt_vector(os.path.join(train_dir, "y.txt"))

    # Handcrafted Features
    features = ["cksnap", "mismatch", "rckmer", "psetnc", "tpcp"]
    for i, feature in enumerate(features):
        raw_train[f'rf_{i}'] = load_whitespace_matrix(os.path.join(train_dir, f"{feature}.txt"))

    # Sequence Features
    raw_train['w2v'] = load_whitespace_matrix(os.path.join(train_dir, "word2vec.txt"))
    # 尝试加载 DNABERT (如果不存在则忽略)
    bert_path = os.path.join(train_dir, "dnabert_features.txt")
    if os.path.exists(bert_path):
        raw_train['bert'] = load_whitespace_matrix(bert_path)
    else:
        print("[Info] 未找到 DNABERT 特征，将跳过。")
        raw_train['bert'] = None

    # === 核心修复: 自动对齐 Train 数据长度 ===
    aligned_train = align_and_truncate(raw_train)

    # 重新组装 Train 张量
    y_train = aligned_train['y']

    # 组装 RF 特征
    rf_list = [aligned_train[f'rf_{i}'] for i in range(len(features))]
    x_train_rf = torch.cat(rf_list, dim=1)

    # 组装 Seq 特征
    if aligned_train['bert'] is not None:
        x_train_seq = torch.cat([aligned_train['w2v'], aligned_train['bert']], dim=1)
    else:
        x_train_seq = aligned_train['w2v']

    # ================= 2. 加载所有数据 (Test) =================
    print("\n--- Loading Test Data ---")
    raw_test = {}

    raw_test['y'] = load_txt_vector(os.path.join(test_dir, "y.txt"))
    for i, feature in enumerate(features):
        raw_test[f'rf_{i}'] = load_whitespace_matrix(os.path.join(test_dir, f"{feature}.txt"))

    raw_test['w2v'] = load_whitespace_matrix(os.path.join(test_dir, "word2vec.txt"))
    bert_path_test = os.path.join(test_dir, "dnabert_features.txt")
    if os.path.exists(bert_path_test):
        raw_test['bert'] = load_whitespace_matrix(bert_path_test)
    else:
        raw_test['bert'] = None

    # === 核心修复: 自动对齐 Test 数据长度 ===
    aligned_test = align_and_truncate(raw_test)

    y_test = aligned_test['y']
    rf_list_test = [aligned_test[f'rf_{i}'] for i in range(len(features))]
    x_test_rf = torch.cat(rf_list_test, dim=1)

    if aligned_test['bert'] is not None:
        x_test_seq = torch.cat([aligned_test['w2v'], aligned_test['bert']], dim=1)
    else:
        x_test_seq = aligned_test['w2v']

    # ================= 3. 归一化和处理 =================
    print("\n--- Normalizing ---")
    # 归一化 RF 特征
    x_train_rf_norm = normalization_minmax(x_train_rf)
    # 使用训练集的 min/max 归一化测试集
    min_v = x_train_rf.min(0).values
    max_v = x_train_rf.max(0).values
    x_test_rf_norm = (x_test_rf - min_v) / (max_v - min_v + 1e-8)

    print(f"[Data Check] Train Samples: {len(y_train)}")
    print(f"[Data Check] Test Samples:  {len(y_test)}")
    print(f"[Data Check] RF Dim: {x_train_rf.shape[1]}, Seq Dim: {x_train_seq.shape[1]}")

    # ================= 4. 划分权重集 (Weight Set) =================
    (x_rf_tr, x_rf_w,
     x_seq_tr, x_seq_w,
     y_tr, y_w) = stratified_split_multi(
        x_train_rf_norm, x_train_seq, y_train, test_size=1 / 8, seed=10
    )

    print(f"Train Splitted: {x_rf_tr.shape[0]}, Weight Set: {x_rf_w.shape[0]}")

    # --- Step 1: Random Forest ---
    print("\n--- Running Random Forest ---")
    rf_weight_proba, rf_test_proba, rf_test_class = RF.pred(
        x_rf_tr, y_tr, x_rf_w, x_test_rf_norm, n_trees=300
    )
    EvaluateMetrics(y_test, rf_test_class, rf_test_proba, tag="RF")

    # --- Step 2: Hybrid Transformer ---
    print("\n--- Running Hybrid Transformer ---")
    cnn_weight_proba, cnn_test_proba, cnn_test_class = run_hybrid_training(
        X_seq_tr=x_seq_tr, X_feat_tr=x_rf_tr, y_tr=y_tr,
        X_seq_w=x_seq_w, X_feat_w=x_rf_w,
        X_seq_te=x_test_seq, X_feat_te=x_test_rf_norm
    )
    EvaluateMetrics(y_test, cnn_test_class, cnn_test_proba, tag="Hybrid-Trans")

    # --- Step 3: Fusion ---
    print("\n--- Running Fusion ---")

    def to_cpu(x):
        return x.detach().cpu() if isinstance(x, torch.Tensor) else torch.tensor(x)

    proba, label = Weighted_average_trans.weight(
        to_cpu(y_w),
        to_cpu(rf_weight_proba), to_cpu(cnn_weight_proba),
        to_cpu(rf_test_proba), to_cpu(cnn_test_proba)
    )
    EvaluateMetrics(y_test, label, proba, tag="Fusion")


if __name__ == "__main__":
    main()