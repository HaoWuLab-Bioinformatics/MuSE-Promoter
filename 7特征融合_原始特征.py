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

ROOT = os.path.dirname(os.path.abspath(__file__))


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
    if os.path.exists(path):
        return path
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
# 1. 你的原始模型架构 (保持完全不变)
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 15000):  # 稍微调大max_len以容纳拼接后的长度
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

        # --- Stream A: 序列处理 (Combined Sequence: Word2Vec + DNABERT) ---
        self.input_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32)
        )

        # 2. 多尺度 CNN 编码器
        self.cnn_encoder = nn.Sequential(
            # Stage 1
            MultiScaleCNN(32, 64),
            nn.MaxPool1d(4),

            SEBlock(64),

            # Stage 2
            nn.Conv1d(64, d_model, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(4),

            SEBlock(d_model)
        )

        # 3. Transformer
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Stream B: 手工特征处理 ---
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
        # x_seq: (N, Combined_Len) -> Word2Vec + DNABERT
        # x_feat: (N, Feat_Dim) -> Handcrafted

        # Stream A
        x = x_seq.unsqueeze(-1)  # (N, L, 1)
        x = self.input_proj(x)  # (N, L, 32)
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
    训练混合模型 (Stream A输入为 Word2Vec+DNABERT 拼接)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hybrid] Training on {device}. Combined Sequence Dim: {X_seq_tr.shape[1]}")

    # 1. 格式转换
    X_seq_tr = torch.as_tensor(X_seq_tr, dtype=torch.float32).to(device)
    X_feat_tr = torch.as_tensor(X_feat_tr, dtype=torch.float32).to(device)
    y_tr = torch.as_tensor(y_tr, dtype=torch.float32).view(-1, 1).to(device)

    # 验证集
    X_seq_val = torch.as_tensor(X_seq_w, dtype=torch.float32).to(device)
    X_feat_val = torch.as_tensor(X_feat_w, dtype=torch.float32).to(device)

    # 划分训练集和内部验证集
    n_total = X_seq_tr.size(0)
    n_val = int(n_total * 0.1)
    perm = torch.randperm(n_total)

    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    train_ds = TensorDataset(X_seq_tr[train_idx], X_feat_tr[train_idx], y_tr[train_idx])
    val_ds = TensorDataset(X_seq_tr[val_idx], X_feat_tr[val_idx], y_tr[val_idx])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    # 2. 模型初始化 (自动适应拼接后的长度)
    current_seq_len = X_seq_tr.shape[1]
    feat_dim = X_feat_tr.shape[1]

    model = HybridTransformer(
        seq_len=current_seq_len,
        feat_dim=feat_dim,
        d_model=128, nhead=4, num_layers=2, dropout=0.3
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    # 3. 训练循环
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

    print(f"[Hybrid] Best Val Loss: {best_loss:.4f} (Stopped at epoch {epoch})")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # 4. 预测
    def predict(x_s, x_f):
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
    path = must_exist(path)
    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            try:
                feat = [float(x) for x in row[1:] if x.strip() != '']
                if feat: rows.append(feat)
            except ValueError:
                continue
    return torch.tensor(rows, dtype=torch.float32)


def load_whitespace_matrix(path):
    """
    自适应读取：
    1. 自动扫描全文件，找到最长的一行。
    2. 将短于最长行的所有行，自动补 0 对齐。
    3. 解决 'ValueError: expected sequence of length...' 问题。
    """
    path = must_exist(path)
    raw_data = []
    max_len = 0

    # 步骤 1: 读取所有行，并记录最大长度
    with open(path, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                parts = [float(x) for x in line.split()]
                current_len = len(parts)
                if current_len > 0:
                    raw_data.append(parts)
                    if current_len > max_len:
                        max_len = current_len
            except ValueError:
                continue

    if not raw_data:
        raise ValueError(f"文件 {path} 为空或无法解析！")

    print(f"  -> 文件 {os.path.basename(path)}: 自动检测最大长度为 {max_len}，正在对齐...")

    # 步骤 2: 对齐数据 (Padding)
    # 这一步是为了防止锯齿状数组 (Jagged Array) 导致的报错
    for i in range(len(raw_data)):
        curr_len = len(raw_data[i])
        if curr_len < max_len:
            # 补零
            raw_data[i].extend([0.0] * (max_len - curr_len))
        # 注意：这里我们只做补零，不主动截断，保留数据的完整性

    return torch.tensor(raw_data, dtype=torch.float32)
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

    idx0 = idx0[torch.randperm(idx0.numel(), generator=g)]
    idx1 = idx1[torch.randperm(idx1.numel(), generator=g)]

    n0 = int(len(idx0) * test_size)
    n1 = int(len(idx1) * test_size)

    w_idx = torch.cat([idx0[:n0], idx1[:n1]])
    tr_idx = torch.cat([idx0[n0:], idx1[n1:]])

    w_idx = w_idx[torch.randperm(len(w_idx), generator=g)]
    tr_idx = tr_idx[torch.randperm(len(tr_idx), generator=g)]

    return X1[tr_idx], X1[w_idx], X2[tr_idx], X2[w_idx], y[tr_idx], y[w_idx]


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().view(-1).numpy()
    elif isinstance(x, np.ndarray):
        return x.flatten()
    else:
        return np.array(x).flatten()


def EvaluateMetrics(y_test, y_pred_class, y_pred_prob, tag=""):
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

def load_whitespace_matrix_keep_rows(path):
    """
    更稳健读取：
    - 不再因为坏行/空行而“丢行”，避免不同特征行数不一致
    - 先通读找 max_len，再逐行解析；解析失败或空行用全0补齐
    """
    path = must_exist(path)

    # 第1遍：找 max_len（忽略完全无法解析的行，但不统计为有效）
    max_len = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts = [float(x) for x in line.split()]
                if len(parts) > max_len:
                    max_len = len(parts)
            except ValueError:
                continue

    if max_len == 0:
        raise ValueError(f"文件 {path} 没有任何可解析的数值行（max_len=0）")

    print(f"  -> 文件 {os.path.basename(path)}: 自动检测最大长度为 {max_len}，正在对齐(keep_rows)...")

    # 第2遍：逐行解析，失败就补0（不丢行）
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                # 空行也占位（保持行数一致）
                rows.append([0.0] * max_len)
                continue
            try:
                parts = [float(x) for x in line.split()]
                if len(parts) < max_len:
                    parts.extend([0.0] * (max_len - len(parts)))
                elif len(parts) > max_len:
                    parts = parts[:max_len]
                rows.append(parts)
            except ValueError:
                # 坏行占位
                rows.append([0.0] * max_len)

    return torch.tensor(rows, dtype=torch.float32)

def main():
    set_seed(2025)

    # -----------------------------------
    cell_name = "E.coli"
    # -----------------------------------

    print(f"Loading data for cell line: {cell_name} ...")

    # 1. 加载标签
    y_train = load_txt_vector(pjoin("feature", cell_name, "train", "y.txt"))
    y_test = load_txt_vector(pjoin("feature", cell_name, "test", "y.txt"))

    # 2. 加载 Handcrafted Features (5种) -> Stream B
    features = ["cksnap", "mismatch", "rckmer", "psetnc", "tpcp"]
    x_train_rf = []
    x_test_rf = []

    for feature in features:
        f_tr = load_whitespace_matrix_keep_rows(pjoin("feature", cell_name, "train", f"{feature}.txt"))
        f_te = load_whitespace_matrix_keep_rows(pjoin("feature", cell_name, "test", f"{feature}.txt"))

        # 训练/测试列数对齐（以训练为准）
        if f_tr.shape[1] != f_te.shape[1]:
            target_dim = f_tr.shape[1]
            if f_te.shape[1] < target_dim:
                pad = torch.zeros((f_te.shape[0], target_dim - f_te.shape[1]))
                f_te = torch.cat([f_te, pad], dim=1)
            else:
                f_te = f_te[:, :target_dim]

        x_train_rf.append(f_tr)
        x_test_rf.append(f_te)

    x_train_rf = torch.cat(x_train_rf, dim=1)
    x_test_rf = torch.cat(x_test_rf, dim=1)

    # 归一化 RF 特征
    x_train_rf_norm = normalization_minmax(x_train_rf)
    x_test_rf_norm = (x_test_rf - x_train_rf.min(0).values) / (
            x_train_rf.max(0).values - x_train_rf.min(0).values + 1e-8)

    # ... (前面的代码不变) ...

    # 3. 加载 Word2Vec (使用新的自适应函数)
    print("Loading Word2Vec features...")
    x_train_w2v = load_whitespace_matrix(pjoin("feature", cell_name, "train", "word2vec.txt"))
    x_test_w2v = load_whitespace_matrix(pjoin("feature", cell_name, "test", "word2vec.txt"))

    # ---【关键步骤】维度安全检查与对齐 ---
    # 确保测试集的特征维度和训练集一致，不够的补0，多了的截断
    if x_train_w2v.shape[1] != x_test_w2v.shape[1]:
        print(f"[Warning] 维度不一致! Train: {x_train_w2v.shape[1]}, Test: {x_test_w2v.shape[1]}. 正在强制对齐 Test...")
        target_dim = x_train_w2v.shape[1]
        current_test_dim = x_test_w2v.shape[1]

        if current_test_dim < target_dim:
            # 测试集维度不够，补 0
            padding = torch.zeros((x_test_w2v.shape[0], target_dim - current_test_dim))
            x_test_w2v = torch.cat([x_test_w2v, padding], dim=1)
        else:
            # 测试集维度多了，截断
            x_test_w2v = x_test_w2v[:, :target_dim]
    # ------------------------------------

    # 4. 加载 DNABERT (同样自适应)
    print("Loading DNABERT features...")
    x_train_bert = load_whitespace_matrix(pjoin("feature", cell_name, "train", "dnabert_features.txt"))
    x_test_bert = load_whitespace_matrix(pjoin("feature", cell_name, "test", "dnabert_features.txt"))

    # --- DNABERT 维度对齐检查 ---
    if x_train_bert.shape[1] != x_test_bert.shape[1]:
        target_dim = x_train_bert.shape[1]
        if x_test_bert.shape[1] < target_dim:
            padding = torch.zeros((x_test_bert.shape[0], target_dim - x_test_bert.shape[1]))
            x_test_bert = torch.cat([x_test_bert, padding], dim=1)
        else:
            x_test_bert = x_test_bert[:, :target_dim]

    # ... (后面的代码不变) ...
    # 5. 拼接序列特征 -> Stream A
    # 将 Word2Vec (N, 8000) 和 DNABERT (N, 768) 拼在一起 -> (N, 8768)
    print("Concatenating Sequence Features (Word2Vec + DNABERT)...")
    x_train_seq = torch.cat([x_train_w2v, x_train_bert], dim=1)
    x_test_seq = torch.cat([x_test_w2v, x_test_bert], dim=1)

    print(f"[Data Check] Combined Sequence Dim: {x_train_seq.shape}, RF Features Dim: {x_train_rf.shape}")

    # 6. 划分权重集 (Weight Set)
    (x_rf_tr, x_rf_w,
     x_seq_tr, x_seq_w,
     y_tr, y_w) = stratified_split_multi(
        x_train_rf_norm, x_train_seq, y_train, test_size=1 / 8, seed=10
    )

    print(f"Train size: {x_rf_tr.shape[0]}, Weight size: {x_rf_w.shape[0]}, Test size: {x_test_rf.shape[0]}")

    # --- Step 1: Random Forest (Baseline) ---
    print("\n--- Running Random Forest (on Handcrafted Features) ---")
    rf_weight_proba, rf_test_proba, rf_test_class = RF.pred(
        x_rf_tr, y_tr, x_rf_w, x_test_rf_norm, n_trees=300
    )
    EvaluateMetrics(y_test, rf_test_class, rf_test_proba, tag="RF")

    # --- Step 2: Hybrid Transformer ---
    # 输入为: Stream A (Word2Vec+BERT) 和 Stream B (Handcrafted)
    print("\n--- Running Hybrid Transformer ---")
    cnn_weight_proba, cnn_test_proba, cnn_test_class = run_hybrid_training(
        X_seq_tr=x_seq_tr, X_feat_tr=x_rf_tr, y_tr=y_tr,
        X_seq_w=x_seq_w, X_feat_w=x_rf_w,
        X_seq_te=x_test_seq, X_feat_te=x_test_rf_norm
    )
    EvaluateMetrics(y_test, cnn_test_class, cnn_test_proba, tag="Hybrid-Trans")

    # --- Step 3: Fusion ---
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