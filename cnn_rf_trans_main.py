import os
import csv
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 假设这些模块你本地有，保持引用
import Weighted_average_trans
import RF

# ==========================================
# 1. 配置与工具函数
# ==========================================
ROOT = os.path.dirname(os.path.abspath(__file__))


def pjoin(*parts):
    return os.path.join(ROOT, *parts)


def set_seed(seed: int = 10):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def must_exist(path: str, hint: str = ""):
    if os.path.exists(path):
        return path
    # 尝试忽略大小写查找
    d = os.path.dirname(path)
    base = os.path.basename(path)
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.lower() == base.lower():
                return os.path.join(d, fn)
    msg = f"[FileMissing] 找不到文件：{path}"
    if hint: msg += f"\n提示：{hint}"
    raise FileNotFoundError(msg)


# ==========================================
# 2. 模型定义 (Hybrid Transformer)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
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
        # x: (B, L, D)
        L = x.size(1)
        if L > self.pe.size(1):
            self.pe = self._build_pe(L, self.d_model).to(x.device)
        return x + self.pe[:, :L, :]


class HybridTransformer(nn.Module):
    def __init__(self, seq_len=8000, feat_dim=1258, vocab_size=30, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()

        # --- Stream A: 序列处理 (Sequence) ---
        # 1. Embedding: 把整数索引变成向量
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)

        # 2. CNN: 提取局部特征并降维 (8000 -> 500)
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),  # 8000 -> 2000

            nn.Conv1d(64, d_model, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(d_model), nn.ReLU(),
            nn.MaxPool1d(4)  # 2000 -> 500
        )

        # 3. Transformer
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Stream B: 手工特征处理 (Features) ---
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU()
        )

        # --- Fusion ---
        fusion_dim = d_model + 128
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq, x_feat):
        # x_seq: (N, 8000) LongTensor
        # x_feat: (N, 1258) FloatTensor

        # Stream A
        x = self.embedding(x_seq)  # (N, 8000, 32)
        x = x.permute(0, 2, 1)  # (N, 32, 8000)
        x = self.cnn_encoder(x)  # (N, d_model, 500)
        x = x.permute(0, 2, 1)  # (N, 500, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        seq_emb = x.mean(dim=1)  # (N, d_model)

        # Stream B
        feat_emb = self.feat_mlp(x_feat)  # (N, 128)

        # Combine
        combined = torch.cat([seq_emb, feat_emb], dim=1)
        return self.classifier(combined)


# ==========================================
# 3. 训练流程封装
# ==========================================
def run_hybrid_training(
        X_seq_tr, X_feat_tr, y_tr,  # Train
        X_seq_w, X_feat_w,  # Weight Set (Validation for training, then predict)
        X_seq_te, X_feat_te,  # Test
        vocab_size=30
):
    """
    训练混合模型并返回预测结果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hybrid] Training on {device}...")

    # 1. 格式转换
    # 序列转 Long, 特征转 Float
    X_seq_tr = torch.as_tensor(X_seq_tr, dtype=torch.long).to(device)
    X_feat_tr = torch.as_tensor(X_feat_tr, dtype=torch.float32).to(device)
    y_tr = torch.as_tensor(y_tr, dtype=torch.float32).view(-1, 1).to(device)

    # 验证集 (使用 Weight Set 做验证，防止过拟合)
    X_seq_val = torch.as_tensor(X_seq_w, dtype=torch.long).to(device)
    X_feat_val = torch.as_tensor(X_feat_w, dtype=torch.float32).to(device)
    # 这里我们没有传入 y_w，所以验证集暂时只用来监控 Loss，或者需要 y_weight
    # 为了简单，我们把 X_seq_tr 划分一部分出来做验证，或者直接用 Training Error 监控（不推荐）
    # *改进*：从 Training Set 中再划出 10% 做内部验证

    n_total = X_seq_tr.size(0)
    n_val = int(n_total * 0.1)
    perm = torch.randperm(n_total)

    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    train_ds = TensorDataset(X_seq_tr[train_idx], X_feat_tr[train_idx], y_tr[train_idx])
    val_ds = TensorDataset(X_seq_tr[val_idx], X_feat_tr[val_idx], y_tr[val_idx])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    # 2. 模型初始化
    feat_dim = X_feat_tr.shape[1]
    model = HybridTransformer(
        seq_len=8000, feat_dim=feat_dim, vocab_size=vocab_size,
        d_model=128, nhead=4, num_layers=2, dropout=0.3
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    # 3. 训练循环
    best_loss = float('inf')
    best_state = None
    patience = 10
    bad = 0

    for epoch in range(150):  # Max Epochs
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

        # Validation
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

    print(f"[Hybrid] Best Val Loss: {best_loss:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # 4. 预测 (Weight Set 和 Test Set)
    def predict(x_s, x_f):
        ds = TensorDataset(
            torch.as_tensor(x_s, dtype=torch.long),
            torch.as_tensor(x_f, dtype=torch.float32)
        )
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
# 4. 数据加载与处理
# ==========================================
def load_txt_vector(path):
    path = must_exist(path)
    vals = []
    with open(path, "r") as f:
        for line in f:
            if line.strip(): vals.append(float(line.strip()))
    return torch.tensor(vals, dtype=torch.float32)


def load_csv_matrix_skip_first_col(path):
    path = must_exist(path)
    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            # 跳过第一列ID，尝试转换后续列
            try:
                # 过滤空字符串
                feat = [float(x) for x in row[1:] if x.strip() != '']
                if feat: rows.append(feat)
            except ValueError:
                continue
    return torch.tensor(rows, dtype=torch.float32)


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
    return torch.tensor(rows, dtype=torch.float32)  # 注意：这里暂时读为float，后面会转long


def normalization_minmax(data):
    minv = data.min(dim=0).values
    maxv = data.max(dim=0).values
    rng = maxv - minv
    rng[rng == 0] = 1.0
    return (data - minv) / rng


def stratified_split_multi(X1, X2, y, test_size=1 / 8, seed=10):
    g = torch.Generator().manual_seed(seed)
    y_int = y.long().view(-1)

    idx0 = torch.nonzero(y_int == 0, as_tuple=False).view(-1)
    idx1 = torch.nonzero(y_int == 1, as_tuple=False).view(-1)

    # Shuffle
    idx0 = idx0[torch.randperm(idx0.numel(), generator=g)]
    idx1 = idx1[torch.randperm(idx1.numel(), generator=g)]

    n0 = int(len(idx0) * test_size)
    n1 = int(len(idx1) * test_size)

    w_idx = torch.cat([idx0[:n0], idx1[:n1]])
    tr_idx = torch.cat([idx0[n0:], idx1[n1:]])

    # Reshuffle
    w_idx = w_idx[torch.randperm(len(w_idx), generator=g)]
    tr_idx = tr_idx[torch.randperm(len(tr_idx), generator=g)]

    return X1[tr_idx], X1[w_idx], X2[tr_idx], X2[w_idx], y[tr_idx], y[w_idx]


# ==========================================
# 辅助函数：统一转为一维 Numpy (修复报错的关键)
# ==========================================
def _to_numpy(x):
    """
    安全地将 PyTorch Tensor 或 List 或 Numpy Array 转换为 1D Numpy Array
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().view(-1).numpy()
    elif isinstance(x, np.ndarray):
        return x.flatten()
    else:
        # 处理 list 或其他
        return np.array(x).flatten()


def EvaluateMetrics(y_test, y_pred_class, y_pred_prob, tag=""):
    from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef

    # 1. 统一转为 Numpy，避免 .view(-1) 在 Numpy 数组上报错
    y_true = _to_numpy(y_test)
    y_cls = _to_numpy(y_pred_class)
    y_prob = _to_numpy(y_pred_prob)

    # 2. 计算指标
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_cls)
    mcc = matthews_corrcoef(y_true, y_cls)

    print(f"[{tag}] AUC={auc:.6f} ACC={acc:.6f} MCC={mcc:.6f}")
# ==========================================
# 5. Main Execution
# ==========================================
# 在文件最开头添加这个，强制 CUDA 同步报错，方便调试
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main():
    set_seed(2025)
    cell_lines = "GM12878"

    print(f"Loading data for {cell_lines}...")

    # 1. Labels
    y_train = load_txt_vector(pjoin("data", cell_lines, "train", "y_train.txt"))
    y_test = load_txt_vector(pjoin("data", cell_lines, "test", "y_test.txt"))

    # 2. RF Features
    features = ["cksnap", "mismatch", "rckmer", "psetnc", "tpcp"]
    x_train_rf = []
    x_test_rf = []

    for feature in features:
        f_tr = load_csv_matrix_skip_first_col(pjoin("feature", cell_lines, "train", f"{feature}.csv"))
        f_te = load_csv_matrix_skip_first_col(pjoin("feature", cell_lines, "test", f"{feature}.csv"))
        x_train_rf.append(f_tr)
        x_test_rf.append(f_te)

    x_train_rf = torch.cat(x_train_rf, dim=1)
    x_test_rf = torch.cat(x_test_rf, dim=1)

    # Normalize RF Features
    x_train_rf_norm = normalization_minmax(x_train_rf)
    x_test_rf_norm = (x_test_rf - x_train_rf.min(0).values) / (
            x_train_rf.max(0).values - x_train_rf.min(0).values + 1e-8)

    # 3. Sequence Features (Word2Vec / Raw Seq)
    # [注意] 这里原文件可能是 float，我们需要小心处理
    x_train_seq_raw = load_whitespace_matrix(pjoin("feature", cell_lines, "train", "word2vec.txt"))
    x_test_seq_raw = load_whitespace_matrix(pjoin("feature", cell_lines, "test", "word2vec.txt"))

    # ==========================================
    # [关键修复] 数据清洗与索引修正
    # ==========================================
    # 转换为 Long
    x_train_seq = x_train_seq_raw.long()
    x_test_seq = x_test_seq_raw.long()

    # 1. 检查是否有负数 (Embedding 不接受负数)
    min_val_tr = x_train_seq.min().item()
    min_val_te = x_test_seq.min().item()
    print(f"[Data Check] Min Index Train: {min_val_tr}, Test: {min_val_te}")

    if min_val_tr < 0 or min_val_te < 0:
        print("[Warning] Found negative indices! Clamping to 0 (padding).")
        x_train_seq = torch.clamp(x_train_seq, min=0)
        x_test_seq = torch.clamp(x_test_seq, min=0)

    # 2. 计算 Vocab Size
    max_idx = max(x_train_seq.max().item(), x_test_seq.max().item())

    # 你的数据最大值是 2.44，转 long 后是 2。
    # vocab_size 至少要是 max_idx + 1。为了安全，我们设大一点，或者如果有特殊的 padding 需求
    real_vocab_size = int(max_idx) + 10

    print(f"[Data Check] Max Index in Sequence (after long): {max_idx}")
    print(f"[Config] Setting vocab_size to: {real_vocab_size}")
    # ==========================================

    # 4. Split Weight Set
    (x_rf_tr, x_rf_w,
     x_seq_tr, x_seq_w,
     y_tr, y_w) = stratified_split_multi(
        x_train_rf_norm, x_train_seq, y_train, test_size=1 / 8, seed=10
    )

    print(f"Train size: {x_rf_tr.shape[0]}, Weight size: {x_rf_w.shape[0]}, Test size: {x_test_rf.shape[0]}")

    # --- RF ---
    print("\n--- Running Random Forest ---")
    rf_weight_proba, rf_test_proba, rf_test_class = RF.pred(
        x_rf_tr, y_tr, x_rf_w, x_test_rf_norm, n_trees=300
    )
    EvaluateMetrics(y_test, rf_test_class, rf_test_proba, tag="RF")

    # --- Hybrid Transformer ---
    print("\n--- Running Hybrid Transformer ---")
    # [调试] 打印一下输入数据的形状，确保没有维度错误
    print(f"[Debug] Hybrid Input Shapes: Seq={x_seq_tr.shape}, Feat={x_rf_tr.shape}")

    cnn_weight_proba, cnn_test_proba, cnn_test_class = run_hybrid_training(
        X_seq_tr=x_seq_tr, X_feat_tr=x_rf_tr, y_tr=y_tr,
        X_seq_w=x_seq_w, X_feat_w=x_rf_w,
        X_seq_te=x_test_seq, X_feat_te=x_test_rf_norm,
        vocab_size=real_vocab_size
    )
    EvaluateMetrics(y_test, cnn_test_class, cnn_test_proba, tag="Hybrid-Trans")

    # --- Fusion ---
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
