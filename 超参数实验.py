import os

# ==========================================
# 1. 环境配置 (必须放在最前面)
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import csv
import random
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score

# ==========================================
# 假设这些模块你本地有 (请确保在同一目录)
import Weighted_average_trans
import RF

# ==========================================

ROOT = os.path.dirname(os.path.abspath(__file__))
# 原代码
# OUTPUT_FILE = "超参数敏感.xlsx"

# 修改为
OUTPUT_FILE = "超参数敏感.csv"
CELL_LINE = "GM12878"  # 当前实验细胞系

# ==========================================
# 实验配置
# ==========================================
# 基准参数 (Baseline)
BASELINE_CONFIG = {
    "d_model": 128,
    "num_layers": 2,
    "nhead": 4,
    "out_channels": 64,
    "learning_rate": 2e-4,
    "dropout": 0.2,
    "n_trees": 300
}

# 实验列表
EXPERIMENTS = [
    ("d_model", [64, 128, 256, 512]),
    ("num_layers", [1, 2, 3, 4]),
    ("nhead", [2, 4, 8]),
    ("out_channels", [32, 64, 128]),
    ("learning_rate", [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]),
    ("dropout", [0.1, 0.2, 0.3, 0.4, 0.5]),
    ("n_trees", [100, 200, 300, 500, 1000]),
]


def save_row_to_csv(row_dict):
    """
    保存结果到 CSV 文件 (比 Excel 更稳定，支持实时追加)
    """
    df_new = pd.DataFrame([row_dict])

    # 检查文件是否存在
    if not os.path.exists(OUTPUT_FILE):
        # 如果文件不存在，写入数据并包含表头 (header=True)
        df_new.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='w')
    else:
        # 如果文件存在，追加数据，不写表头 (header=False)
        df_new.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig', mode='a', header=False)

    print(f"  [Saved] Results updated to {OUTPUT_FILE}")
# ==========================================
# 工具函数
# ==========================================
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
    if os.path.exists(path): return path
    raise FileNotFoundError(f"File not found: {path}")


def load_txt_vector(path):
    """读取标签文件 (y.txt)"""
    path = must_exist(path)
    vals = []
    with open(path, "r") as f:
        for line in f:
            if line.strip(): vals.append(float(line.strip()))
    return torch.tensor(vals, dtype=torch.float32)


def load_whitespace_matrix_keep_rows(path):
    """
    读取特征矩阵，遇到解析错误行补0，保持行数尽可能完整
    """
    path = must_exist(path)
    # 1. 第一遍扫描：找最大列数 (max_len)
    max_len = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                parts = [float(x) for x in line.split()]
                if len(parts) > max_len: max_len = len(parts)
            except:
                continue

    if max_len == 0: return None

    # 2. 第二遍扫描：读取并对齐
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                rows.append([0.0] * max_len)  # 空行补全0
                continue
            try:
                parts = [float(x) for x in line.split()]
                # 短了补0，长了截断
                if len(parts) < max_len:
                    parts.extend([0.0] * (max_len - len(parts)))
                elif len(parts) > max_len:
                    parts = parts[:max_len]
                rows.append(parts)
            except:
                rows.append([0.0] * max_len)  # 坏行补全0
    return torch.tensor(rows, dtype=torch.float32)


def load_and_align_features(base_path, cell_line, mode, feature_names):
    """
    【核心修复】读取一组特征文件，并强制截断到最小行数，解决对齐报错
    """
    raw_tensors = []
    min_rows = float('inf')

    print(f"  [Load] Loading {mode} features for {cell_line}...")

    # 1. 读取
    for f_name in feature_names:
        path = os.path.join(base_path, cell_line, mode, f"{f_name}.txt")
        tensor = load_whitespace_matrix_keep_rows(path)
        if tensor is None:
            raise ValueError(f"File {path} is empty!")

        raw_tensors.append(tensor)

        if tensor.shape[0] < min_rows:
            min_rows = tensor.shape[0]

    print(f"    -> Row counts detected: {[t.shape[0] for t in raw_tensors]}")
    print(f"    -> Aligning all to {min_rows} rows.")

    # 2. 截断
    aligned_tensors = []
    for t in raw_tensors:
        if t.shape[0] > min_rows:
            aligned_tensors.append(t[:min_rows, :])
        else:
            aligned_tensors.append(t)

    return aligned_tensors, min_rows


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


def get_metrics(y_true, y_pred_class, y_pred_prob):
    def _to_numpy(x):
        return x.detach().cpu().view(-1).numpy() if isinstance(x, torch.Tensor) else np.array(x).flatten()

    y_t = _to_numpy(y_true)
    y_c = _to_numpy(y_pred_class)
    y_p = _to_numpy(y_pred_prob)
    try:
        auc = roc_auc_score(y_t, y_p)
    except:
        auc = 0.5
    acc = accuracy_score(y_t, y_c)
    mcc = matthews_corrcoef(y_t, y_c)
    return auc, acc, mcc


# ==========================================
# 模型定义
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 25000):  # 调大以适应拼接长度
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(1)
        if L > self.pe.shape[1]:
            # 如果输入比预设长，动态重新生成（罕见情况）
            device = x.device
            self.pe = torch.zeros(1, L, self.d_model).to(device)
            # 简化的重新生成，实际运行一般不会超出 25000
        return x + self.pe[:, :L, :].to(x.device)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
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
        # 保证通道数分配正确
        oc = out_channels // 4
        self.conv1 = nn.Conv1d(in_channels, oc, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, oc, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(in_channels, oc, kernel_size=11, padding=5)
        # 剩下的通道给最后一个卷积，处理不能整除的情况
        last_oc = out_channels - 3 * oc
        self.conv4 = nn.Conv1d(in_channels, last_oc, kernel_size=1, padding=0)
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
    def __init__(self, seq_len, feat_dim,
                 d_model, nhead, num_layers, dropout, cnn_out_channels):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(1, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32)
        )

        self.cnn_encoder = nn.Sequential(
            MultiScaleCNN(32, cnn_out_channels),
            nn.MaxPool1d(4),
            SEBlock(cnn_out_channels),

            nn.Conv1d(cnn_out_channels, d_model, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(d_model), nn.ReLU(),
            nn.MaxPool1d(4),
            SEBlock(d_model)
        )

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU()
        )

        fusion_dim = d_model + 128
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Dropout(dropout),
            nn.Linear(fusion_dim, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq, x_feat):
        x = x_seq.unsqueeze(-1)
        x = self.input_proj(x).permute(0, 2, 1)
        x = self.cnn_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        seq_emb = x.mean(dim=1)

        feat_emb = self.feat_mlp(x_feat)
        combined = torch.cat([seq_emb, feat_emb], dim=1)
        return self.classifier(combined)


# ==========================================
# 训练与保存逻辑
# ==========================================
def train_and_eval(config, x_seq_tr, x_rf_tr, y_tr, x_seq_w, x_rf_w, y_w, x_seq_te, x_rf_te):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unpack config
    dm = config['d_model']
    nl = config['num_layers']
    nh = config['nhead']
    oc = config['out_channels']
    lr = config['learning_rate']
    dr = config['dropout']

    def to_dev(x):
        return torch.as_tensor(x, dtype=torch.float32).to(device)

    Xt_s, Xt_f, Yt = to_dev(x_seq_tr), to_dev(x_rf_tr), to_dev(y_tr).view(-1, 1)
    Xw_s, Xw_f, Yw = to_dev(x_seq_w), to_dev(x_rf_w), to_dev(y_w).view(-1, 1)

    train_ds = TensorDataset(Xt_s, Xt_f, Yt)
    val_ds = TensorDataset(Xw_s, Xw_f, Yw)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=128)

    model = HybridTransformer(
        seq_len=Xt_s.shape[1], feat_dim=Xt_f.shape[1],
        d_model=dm, nhead=nh, num_layers=nl, dropout=dr, cnn_out_channels=oc
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    # Early Stopping
    best_loss = float('inf')
    patience = 6  # 稍微减少patience加速实验
    bad = 0
    best_state = None

    for epoch in range(100):
        model.train()
        for b_s, b_f, b_y in train_dl:
            optimizer.zero_grad()
            logits = model(b_s, b_f)
            loss = loss_fn(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_s, b_f, b_y in val_dl:
                logits = model(b_s, b_f)
                val_loss += loss_fn(logits, b_y).item()
        val_loss /= len(val_dl)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience: break

    if best_state: model.load_state_dict(best_state)
    model.eval()

    def predict(xs, xf):
        ds = TensorDataset(torch.as_tensor(xs, dtype=torch.float32), torch.as_tensor(xf, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=128, shuffle=False)
        probs = []
        with torch.no_grad():
            for b_s, b_f in dl:
                b_s, b_f = b_s.to(device), b_f.to(device)
                p = torch.sigmoid(model(b_s, b_f))
                probs.append(p.cpu())
        return torch.cat(probs, dim=0).view(-1, 1)

    w_prob = predict(x_seq_w, x_rf_w)
    t_prob = predict(x_seq_te, x_rf_te)
    t_cls = (t_prob >= 0.5).long()

    return w_prob, t_prob, t_cls


def save_row_to_excel(row_dict):
    df_new = pd.DataFrame([row_dict])
    if os.path.exists(OUTPUT_FILE):
        try:
            with pd.ExcelWriter(OUTPUT_FILE, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
                try:
                    start_row = writer.book["Results"].max_row
                    df_new.to_excel(writer, index=False, header=False, startrow=start_row, sheet_name="Results")
                except KeyError:
                    df_new.to_excel(writer, index=False, sheet_name="Results")
        except Exception as e:
            # 备用方案
            df_old = pd.read_excel(OUTPUT_FILE)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
            df_final.to_excel(OUTPUT_FILE, index=False, sheet_name="Results")
    else:
        df_new.to_excel(OUTPUT_FILE, index=False, sheet_name="Results")
    print(f"  [Saved] Results updated to {OUTPUT_FILE}")


# ==========================================
# Main
# ==========================================
def main():
    set_seed(2025)

    print(f"Loading data for {CELL_LINE} (Once for all experiments)...")
    # 请根据实际路径调整
    base_path = os.path.join(ROOT, "EPfeature")

    rf_feats = ["cksnap", "mismatch", "rckmer", "psetnc", "tpcp"]
    seq_feats = ["word2vec", "dnabert_features"]

    # ---------------------------------------------
    # 1. 加载并对齐 Handcrafted Features (RF)
    # ---------------------------------------------
    tr_rf_list, tr_min = load_and_align_features(base_path, CELL_LINE, "train", rf_feats)
    te_rf_list, te_min = load_and_align_features(base_path, CELL_LINE, "test", rf_feats)

    # ---------------------------------------------
    # 2. 加载标签并对齐
    # ---------------------------------------------
    y_train = load_txt_vector(os.path.join(base_path, CELL_LINE, "train", "y.txt"))
    y_test = load_txt_vector(os.path.join(base_path, CELL_LINE, "test", "y.txt"))

    # 强制截断标签以匹配特征行数
    if y_train.shape[0] > tr_min:
        y_train = y_train[:tr_min]
    if y_test.shape[0] > te_min:
        y_test = y_test[:te_min]

    # ---------------------------------------------
    # 3. 拼接 RF 特征并归一化
    # ---------------------------------------------
    x_train_rf_raw = torch.cat(tr_rf_list, dim=1)
    x_test_rf_raw = torch.cat(te_rf_list, dim=1)

    x_train_rf = normalization_minmax(x_train_rf_raw)
    min_v, max_v = x_train_rf_raw.min(0).values, x_train_rf_raw.max(0).values
    x_test_rf = (x_test_rf_raw - min_v) / (max_v - min_v + 1e-8)

    # ---------------------------------------------
    # 4. 加载并对齐 Sequence Features
    # ---------------------------------------------
    # 注意：Sequence特征的行数必须也截断到 tr_min 和 te_min
    tr_seq_list, _ = load_and_align_features(base_path, CELL_LINE, "train", seq_feats)
    te_seq_list, _ = load_and_align_features(base_path, CELL_LINE, "test", seq_feats)

    # 强制截断行数 (虽然 load_and_align 内部做了一次，但要确保和 RF 的行数一致)
    tr_seq_list = [t[:tr_min] for t in tr_seq_list]
    te_seq_list = [t[:te_min] for t in te_seq_list]

    # 拼接 (Train vs Test 的列数对齐处理)
    x_train_seq = torch.cat(tr_seq_list, dim=1)
    x_test_seq = torch.cat(te_seq_list, dim=1)

    if x_train_seq.shape[1] != x_test_seq.shape[1]:
        print(f"  [Align] Adjusting Seq dim: {x_train_seq.shape[1]} vs {x_test_seq.shape[1]}")
        target_dim = x_train_seq.shape[1]
        if x_test_seq.shape[1] < target_dim:
            padding = torch.zeros(x_test_seq.shape[0], target_dim - x_test_seq.shape[1])
            x_test_seq = torch.cat([x_test_seq, padding], dim=1)
        else:
            x_test_seq = x_test_seq[:, :target_dim]

    # ---------------------------------------------
    # 5. 划分权重集
    # ---------------------------------------------
    # 此时所有 X 和 Y 已经严格对齐
    x_rf_tr, x_rf_w, x_seq_tr, x_seq_w, y_tr, y_w = stratified_split_multi(
        x_train_rf, x_train_seq, y_train, test_size=1 / 8, seed=10
    )

    print("\n[Ready] Data loaded and aligned successfully.")
    print(f"  Train: {x_rf_tr.shape[0]}, Weight: {x_rf_w.shape[0]}, Test: {x_test_rf.shape[0]}")

    # ---------------------------------------------
    # 6. 开始敏感性实验
    # ---------------------------------------------
    exp_count = 0
    total_exps = sum([len(vals) for _, vals in EXPERIMENTS])

    for param_name, param_values in EXPERIMENTS:
        print(f"\n>>> Testing Sensitivity for: {param_name}")

        for val in param_values:
            exp_count += 1
            print(f"  [{exp_count}/{total_exps}] Setting {param_name} = {val}")

            # 配置覆盖
            current_config = BASELINE_CONFIG.copy()
            current_config[param_name] = val

            row_data = {
                "Tested_Param": param_name,
                "Param_Value": val,
                **current_config
            }

            try:
                # A. RF (n_trees support)
                nt = current_config['n_trees']
                rf_w_prob, rf_t_prob, rf_cls = RF.pred(x_rf_tr, y_tr, x_rf_w, x_test_rf, n_trees=nt)
                rf_auc, rf_acc, rf_mcc = get_metrics(y_test, rf_cls, rf_t_prob)

                row_data["RF_AUC"] = rf_auc
                row_data["RF_ACC"] = rf_acc
                row_data["RF_MCC"] = rf_mcc

                # B. Hybrid
                cnn_w_prob, cnn_t_prob, cnn_cls = train_and_eval(
                    current_config,
                    x_seq_tr, x_rf_tr, y_tr,
                    x_seq_w, x_rf_w, y_w,
                    x_test_seq, x_test_rf
                )
                ht_auc, ht_acc, ht_mcc = get_metrics(y_test, cnn_cls, cnn_t_prob)

                row_data["Hybrid_AUC"] = ht_auc
                row_data["Hybrid_ACC"] = ht_acc
                row_data["Hybrid_MCC"] = ht_mcc

                # C. Fusion
                def to_cpu(x):
                    return x.detach().cpu() if isinstance(x, torch.Tensor) else torch.tensor(x)

                fus_prob, fus_lbl = Weighted_average_trans.weight(
                    to_cpu(y_w), to_cpu(rf_w_prob), to_cpu(cnn_w_prob),
                    to_cpu(rf_t_prob), to_cpu(cnn_t_prob)
                )
                fu_auc, fu_acc, fu_mcc = get_metrics(y_test, fus_lbl, fus_prob)

                row_data["Fusion_AUC"] = fu_auc
                row_data["Fusion_ACC"] = fu_acc
                row_data["Fusion_MCC"] = fu_mcc

                save_row_to_csv(row_data)

            except Exception as e:
                print(f"  [ERROR] Experiment failed for {param_name}={val}: {e}")
                row_data["Notes"] = f"Error: {str(e)}"
                save_row_to_csv(row_data)
                torch.cuda.empty_cache()

    print(f"\nAll experiments done. Check {OUTPUT_FILE}")


if __name__ == "__main__":
    main()