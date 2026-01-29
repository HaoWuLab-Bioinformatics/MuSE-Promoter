import os
import csv
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score
from collections import Counter

# ==========================================
# 假设这些模块你本地有，保持引用
import Weighted_average_trans
import RF

# ==========================================

# ================= 配置区域 =================
# 根目录
PROJECT_ROOT = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/EPfeature"

# 细胞系列表
CELL_LINES = ["GM12878", "HeLa-S3", "HUVEC", "K562"]

# 结果输出文件
REALTIME_FILE = "Cross_Cell_Validation_ZCurve.csv"

# 【修改点】原始文件名改为 data.fasta
SEQUENCE_FILENAME = "data.fasta"


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
    if os.path.exists(path):
        return path
    d = os.path.dirname(path)
    b = os.path.basename(path)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower() == b.lower():
                return os.path.join(d, f)
    raise FileNotFoundError(f"[FileMissing] 找不到文件：{path}")


# ==========================================
# Z-Curve 生成逻辑 (针对 FASTA 优化)
# ==========================================
def calculate_zcurve_features(sequence):
    """
    计算单条序列的 Z-curve 3维特征 (x, y, z) 并归一化
    """
    seq = sequence.upper().strip()
    length = len(seq)
    if length == 0:
        return [0.0, 0.0, 0.0]

    # 统计碱基数量
    counts = Counter(seq)
    A = counts.get('A', 0)
    C = counts.get('C', 0)
    G = counts.get('G', 0)
    T = counts.get('T', 0)

    # Z-curve 变换公式
    x = (A + G) - (C + T)
    y = (A + C) - (G + T)
    z = (A + T) - (G + C)

    return [x / length, y / length, z / length]


def read_fasta_sequences(fasta_path):
    """
    读取标准 FASTA 文件，返回序列列表。
    处理多行序列的情况，并保持原始顺序。
    """
    sequences = []
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # 如果之前缓存了序列，说明上一条读完了
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
                # header 行直接跳过，不处理
            else:
                # 序列行，加入缓存
                current_seq.append(line)

        # 处理文件末尾的最后一条序列
        if current_seq:
            sequences.append("".join(current_seq))

    return sequences


def check_and_generate_zcurve(folder_path):
    """
    检查目录下是否有 ZCurve.txt，没有则从 data.fasta 生成
    """
    zcurve_path = os.path.join(folder_path, "ZCurve.txt")
    fasta_path = os.path.join(folder_path, SEQUENCE_FILENAME)

    if os.path.exists(zcurve_path):
        # 简单检查一下文件是否为空
        if os.path.getsize(zcurve_path) > 0:
            print(f"  [Check] Found existing ZCurve.txt in {folder_path}")
            return

    print(f"  [Gen] Generating ZCurve.txt from {SEQUENCE_FILENAME}...")

    # 寻找 fasta 文件
    if not os.path.exists(fasta_path):
        try:
            fasta_path = must_exist(fasta_path)  # 尝试忽略大小写
        except FileNotFoundError:
            raise FileNotFoundError(f"  [Error] Cannot find {SEQUENCE_FILENAME} in {folder_path}")

    # 1. 读取 FASTA
    sequences = read_fasta_sequences(fasta_path)
    if not sequences:
        raise ValueError(f"  [Error] {fasta_path} seems empty or invalid format.")

    print(f"  [Gen] Parsed {len(sequences)} sequences from FASTA.")

    # 2. 计算特征
    z_features = []
    for seq in sequences:
        feat = calculate_zcurve_features(seq)
        z_features.append(feat)

    # 3. 写入文件
    with open(zcurve_path, 'w') as f:
        for row in z_features:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")

    print(f"  [Gen] Successfully created {zcurve_path}.")


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


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'  [EarlyStopping] Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f'  [EarlyStopping] Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class HybridTransformer(nn.Module):
    def __init__(self, seq_len=8000, feat_dim=3, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(1, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32))
        self.cnn_encoder = nn.Sequential(
            MultiScaleCNN(32, 64), nn.MaxPool1d(4), SEBlock(64),
            nn.Conv1d(64, d_model, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.MaxPool1d(4), SEBlock(d_model)
        )
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                                                   dropout=dropout, activation="gelu", batch_first=True,
                                                   norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Z-curve 维度是 3
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64),  # 先升维
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 128), nn.ReLU()
        )
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

    X_seq_tr = torch.as_tensor(X_seq_tr, dtype=torch.float32).to(device)
    X_feat_tr = torch.as_tensor(X_feat_tr, dtype=torch.float32).to(device)
    y_tr = torch.as_tensor(y_tr, dtype=torch.float32).view(-1, 1).to(device)

    X_seq_val = torch.as_tensor(X_seq_w, dtype=torch.float32).to(device)
    X_feat_val = torch.as_tensor(X_feat_w, dtype=torch.float32).to(device)
    y_val = torch.as_tensor(y_w, dtype=torch.float32).view(-1, 1).to(device)

    train_ds = TensorDataset(X_seq_tr, X_feat_tr, y_tr)
    val_ds = TensorDataset(X_seq_val, X_feat_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    feat_dim_input = X_feat_tr.shape[1]

    model = HybridTransformer(seq_len=X_seq_tr.shape[1], feat_dim=feat_dim_input).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    early_stopping = EarlyStopping(patience=5, verbose=True, path='temp_best_model_zcurve.pt')

    EPOCHS = 50

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for b_seq, b_feat, b_y in train_loader:
            opt.zero_grad()
            logits = model(b_seq, b_feat)
            loss = loss_fn(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_seq, b_feat, b_y in val_loader:
                logits = model(b_seq, b_feat)
                loss = loss_fn(logits, b_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print(f"  [Info] Early stopping triggered at epoch {epoch + 1}")
            break

    if os.path.exists('temp_best_model_zcurve.pt'):
        model.load_state_dict(torch.load('temp_best_model_zcurve.pt'))
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

    if os.path.exists('temp_best_model_zcurve.pt'):
        os.remove('temp_best_model_zcurve.pt')

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


def align_and_truncate(data_dict):
    lengths = {k: v.shape[0] for k, v in data_dict.items() if v is not None}
    if not lengths: return {}
    min_len = min(lengths.values())
    aligned_dict = {}
    for k, v in data_dict.items():
        if v is not None:
            aligned_dict[k] = v[:min_len]
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


def calculate_metrics(y_true, y_pred_class, y_pred_prob):
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


def load_data_for_cell(cell_name, sub_folder="train"):
    base_dir = os.path.join(PROJECT_ROOT, cell_name, sub_folder)
    print(f"Loading data from: {base_dir}")

    # --- 步骤1: 检查并从 data.fasta 生成 ZCurve.txt ---
    check_and_generate_zcurve(base_dir)

    # --- 步骤2: 加载数据 ---
    raw_data = {}
    raw_data['y'] = load_txt_vector(os.path.join(base_dir, "y.txt"))

    try:
        raw_data['zcurve'] = load_whitespace_matrix(os.path.join(base_dir, "ZCurve.txt"))
    except FileNotFoundError:
        print(f"  [Critical Error] ZCurve.txt creation failed.")
        raise

    raw_data['w2v'] = load_whitespace_matrix(os.path.join(base_dir, "word2vec.txt"))

    bert_path = os.path.join(base_dir, "dnabert_features.txt")
    if os.path.exists(bert_path):
        raw_data['bert'] = load_whitespace_matrix(bert_path)
    else:
        raw_data['bert'] = None

    aligned = align_and_truncate(raw_data)

    y = aligned['y']
    x_rf = aligned['zcurve']

    if aligned['bert'] is not None:
        x_seq = torch.cat([aligned['w2v'], aligned['bert']], dim=1)
    else:
        x_seq = aligned['w2v']

    return x_rf, x_seq, y


def save_result_realtime(result_dict, filename):
    df = pd.DataFrame([result_dict])
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(filename, index=False, mode='a', header=False, encoding='utf-8-sig')


# ==========================================
# Main Execution
# ==========================================
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main():
    set_seed(2025)

    print(f"Results will be saved to: {REALTIME_FILE}")
    print(f"Looking for: {SEQUENCE_FILENAME} to generate ZCurve features.")

    for train_cell in CELL_LINES:
        print(f"\n##################################################")
        print(f" STARTING TRAIN GROUP (Z-Curve): {train_cell}")
        print(f"##################################################")

        try:
            x_train_rf_raw, x_train_seq, y_train = load_data_for_cell(train_cell, "train")
        except Exception as e:
            print(f"Skipping {train_cell} due to error: {e}")
            continue

        min_v = x_train_rf_raw.min(0).values
        max_v = x_train_rf_raw.max(0).values
        range_v = max_v - min_v
        range_v[range_v == 0] = 1.0
        x_train_rf_norm = (x_train_rf_raw - min_v) / range_v

        (x_rf_tr, x_rf_w, x_seq_tr, x_seq_w, y_tr, y_w) = stratified_split_multi(
            x_train_rf_norm, x_train_seq, y_train, test_size=1 / 8, seed=10
        )

        for test_cell in CELL_LINES:
            print(f"\n>>> Case: Train[{train_cell}] vs Test[{test_cell}]")
            torch.cuda.empty_cache()

            try:
                x_test_rf_raw, x_test_seq, y_test = load_data_for_cell(test_cell, "test")
            except Exception as e:
                print(f"  [Error] Failed to load test data for {test_cell}: {e}")
                continue

            x_test_rf_norm = (x_test_rf_raw - min_v) / (range_v + 1e-8)

            # --- Model 1: Random Forest ---
            print("  Running RF (Z-curve)...")
            rf_weight_proba, rf_test_proba, rf_test_class = RF.pred(
                x_rf_tr, y_tr, x_rf_w, x_test_rf_norm, n_trees=300
            )
            rf_auc, rf_acc, rf_mcc = calculate_metrics(y_test, rf_test_class, rf_test_proba)

            save_result_realtime({
                "Train Cell": train_cell, "Test Cell": test_cell, "Model": "RF_ZCurve",
                "AUC": rf_auc, "ACC": rf_acc, "MCC": rf_mcc
            }, REALTIME_FILE)

            # --- Model 2: Hybrid Transformer ---
            print("  Running Hybrid-Trans (Z-curve)...")
            cnn_weight_proba, cnn_test_proba, cnn_test_class = run_hybrid_training(
                X_seq_tr=x_seq_tr, X_feat_tr=x_rf_tr, y_tr=y_tr,
                X_seq_w=x_seq_w, X_feat_w=x_rf_w, y_w=y_w,
                X_seq_te=x_test_seq, X_feat_te=x_test_rf_norm
            )

            ht_auc, ht_acc, ht_mcc = calculate_metrics(y_test, cnn_test_class, cnn_test_proba)

            save_result_realtime({
                "Train Cell": train_cell, "Test Cell": test_cell, "Model": "Hybrid_ZCurve",
                "AUC": ht_auc, "ACC": ht_acc, "MCC": ht_mcc
            }, REALTIME_FILE)

            # --- Model 3: Fusion ---
            print("  Running Fusion...")

            def to_cpu(x):
                return x.detach().cpu() if isinstance(x, torch.Tensor) else torch.tensor(x)

            fusion_proba, fusion_label = Weighted_average_trans.weight(
                to_cpu(y_w), to_cpu(rf_weight_proba), to_cpu(cnn_weight_proba),
                to_cpu(rf_test_proba), to_cpu(cnn_test_proba)
            )
            fu_auc, fu_acc, fu_mcc = calculate_metrics(y_test, fusion_label, fusion_proba)

            save_result_realtime({
                "Train Cell": train_cell, "Test Cell": test_cell, "Model": "Fusion_ZCurve",
                "AUC": fu_auc, "ACC": fu_acc, "MCC": fu_mcc
            }, REALTIME_FILE)

            print(f"  [Saved] Results for {train_cell}->{test_cell} saved.")

    print(f"\nAll Done. Results saved in {REALTIME_FILE}.")


if __name__ == "__main__":
    main()