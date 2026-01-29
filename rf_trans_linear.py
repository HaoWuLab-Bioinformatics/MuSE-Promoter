import os
import csv
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import matthews_corrcoef
# 假设这些模块你本地有，保持引用
import Weighted_average_trans
import RF
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
        # 分支1: 小感受野 (Kernel=3)
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        # 分支2: 中感受野 (Kernel=7)
        self.conv2 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        # 分支3: 大感受野 (Kernel=11)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=11, padding=5)
        # 分支4: 1x1 卷积 (保留原特征)
        self.conv4 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(x)
        out = torch.cat([c1, c2, c3, c4], dim=1)  # 拼接通道
        return self.relu(self.bn(out))
class HybridTransformer(nn.Module):
    def __init__(self, seq_len=8000, feat_dim=1258, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()

        # --- Stream A: 序列处理 (Sequence) ---
        # 1. 增强的输入投影 (MLP)
        self.input_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32)
        )

        # 2. 多尺度 CNN 编码器
        # 这里我们用两个多尺度块，替代原来的普通 CNN
        self.cnn_encoder = nn.Sequential(
            # Stage 1: Multi-Scale Extraction
            MultiScaleCNN(32, 64),
            nn.MaxPool1d(4),  # 8000 -> 2000

            # Attention: SE-Block 增强关键通道
            SEBlock(64),

            # Stage 2: Standard Conv to adjust dimension
            nn.Conv1d(64, d_model, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 2000 -> 500

            # Attention again
            SEBlock(d_model)
        )

        # 3. Transformer (保持不变)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Stream B: 手工特征处理 (保持不变) ---
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
        # x_seq: (N, 8000)
        # x_feat: (N, 1258)

        # Stream A
        x = x_seq.unsqueeze(-1)  # (N, L, 1)
        x = self.input_proj(x)  # (N, L, 32) -> 增强了非线性特征提取
        x = x.permute(0, 2, 1)  # (N, 32, L)

        x = self.cnn_encoder(x)  # (N, d_model, L_reduced)

        x = x.permute(0, 2, 1)  # (N, L_reduced, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        seq_emb = x.mean(dim=1)

        # Stream B
        feat_emb = self.feat_mlp(x_feat)

        # Combine
        combined = torch.cat([seq_emb, feat_emb], dim=1)
        return self.classifier(combined)
def run_hybrid_training(
        X_seq_tr, X_feat_tr, y_tr,  # Train
        X_seq_w, X_feat_w,  # Weight Set (Validation)
        X_seq_te, X_feat_te  # Test
):
    """
    训练混合模型并返回预测结果 (处理连续序列数据)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hybrid] Training on {device} (Mode: Continuous Input)...")

    # 1. 格式转换 [关键修改: 全部转为 float32]
    X_seq_tr = torch.as_tensor(X_seq_tr, dtype=torch.float32).to(device)
    X_feat_tr = torch.as_tensor(X_feat_tr, dtype=torch.float32).to(device)
    y_tr = torch.as_tensor(y_tr, dtype=torch.float32).view(-1, 1).to(device)

    # 验证集
    X_seq_val = torch.as_tensor(X_seq_w, dtype=torch.float32).to(device)
    X_feat_val = torch.as_tensor(X_feat_w, dtype=torch.float32).to(device)

    # 划分训练集和内部验证集 (10% Split)
    n_total = X_seq_tr.size(0)
    n_val = int(n_total * 0.1)
    perm = torch.randperm(n_total)

    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    train_ds = TensorDataset(X_seq_tr[train_idx], X_feat_tr[train_idx], y_tr[train_idx])
    val_ds = TensorDataset(X_seq_tr[val_idx], X_feat_tr[val_idx], y_tr[val_idx])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    # 2. 模型初始化 [关键修改: 移除 vocab_size]
    feat_dim = X_feat_tr.shape[1]
    model = HybridTransformer(
        seq_len=8000, feat_dim=feat_dim,
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
        # [关键修改] 预测输入也必须转为 float32
        ds = TensorDataset(
            torch.as_tensor(x_s, dtype=torch.float32),
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
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def main():
    set_seed(2025)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # ==========================================
    # 【关键修改】在这里填写你的细胞系文件夹名字
    # 例如你的路径是 feature/GM12878/train/...
    # 这里就填 "GM12878"
    cell_name = "mouse nonTATA"
    # ==========================================

    print(f"Loading data for cell line: {cell_name} ...")

    # 1. 加载标签 (假设 data 文件夹结构与 feature 保持一致)
    # 路径变为: data/GM12878/train/y_train.txt
    y_train = load_txt_vector(pjoin("data", cell_name, "train", "y_train.txt"))
    y_test = load_txt_vector(pjoin("data", cell_name, "test", "y_test.txt"))

    # 2. 加载 RF 特征
    features = ["cksnap", "mismatch", "rckmer", "psetnc", "tpcp"]
    x_train_rf = []
    x_test_rf = []

    for feature in features:
        # 路径变为: feature/GM12878/train/cksnap.csv
        f_tr = load_csv_matrix_skip_first_col(pjoin("feature", cell_name, "train", f"{feature}.csv"))
        f_te = load_csv_matrix_skip_first_col(pjoin("feature", cell_name, "test", f"{feature}.csv"))
        x_train_rf.append(f_tr)
        x_test_rf.append(f_te)

    x_train_rf = torch.cat(x_train_rf, dim=1)
    x_test_rf = torch.cat(x_test_rf, dim=1)

    # 归一化
    x_train_rf_norm = normalization_minmax(x_train_rf)
    x_test_rf_norm = (x_test_rf - x_train_rf.min(0).values) / (
            x_train_rf.max(0).values - x_train_rf.min(0).values + 1e-8)

    # 3. 加载 Sequence 特征 (Word2Vec)
    # 路径变为: feature/GM12878/train/word2vec.txt
    x_train_seq = load_whitespace_matrix(pjoin("feature", cell_name, "train", "word2vec.txt"))
    x_test_seq = load_whitespace_matrix(pjoin("feature", cell_name, "test", "word2vec.txt"))

    print(f"[Data Check] Sequence Range: [{x_train_seq.min():.4f}, {x_train_seq.max():.4f}]")

    # 4. 划分权重集 (Weight Set)
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

    # --- Hybrid Transformer (Continuous) ---
    print("\n--- Running Hybrid Transformer (Continuous) ---")

    cnn_weight_proba, cnn_test_proba, cnn_test_class = run_hybrid_training(
        X_seq_tr=x_seq_tr, X_feat_tr=x_rf_tr, y_tr=y_tr,
        X_seq_w=x_seq_w, X_feat_w=x_rf_w,
        X_seq_te=x_test_seq, X_feat_te=x_test_rf_norm
    )
    EvaluateMetrics(y_test, cnn_test_class, cnn_test_proba, tag="Hybrid-Trans")

    # --- Fusion ---
    print("\n--- Running Fusion ---")

    def to_cpu(x): return x.detach().cpu() if isinstance(x, torch.Tensor) else torch.tensor(x)

    proba, label = Weighted_average_trans.weight(
        to_cpu(y_w),
        to_cpu(rf_weight_proba), to_cpu(cnn_weight_proba),
        to_cpu(rf_test_proba), to_cpu(cnn_test_proba)
    )
    EvaluateMetrics(y_test, label, proba, tag="Fusion")

if __name__ == "__main__":
    main()