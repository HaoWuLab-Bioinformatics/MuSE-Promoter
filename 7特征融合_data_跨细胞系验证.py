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

# ==========================================
# 假设这些模块你本地有，保持引用
import Weighted_average_trans
import RF

# ==========================================

# ================= 配置区域 =================
# 1. 特征根目录
FEATURE_ROOT = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/feature"

# 2. 标签/数据根目录
DATA_ROOT = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/data"

# 定义所有需要验证的细胞系
CELL_LINES = ["GM12878", "HeLa-S3", "HUVEC", "K562"]

# 输出文件名
OUTPUT_FILE = "Cross_Cell_Validation_Realtime_Fixed_v2.csv"


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
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class HybridTransformer(nn.Module):
    def __init__(self, seq_len=8000, feat_dim=1258, d_model=128, nhead=4, num_layers=2, dropout=0.2):
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

    model = HybridTransformer(seq_len=X_seq_tr.shape[1], feat_dim=X_feat_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    early_stopping = EarlyStopping(patience=5, verbose=False, path='temp_best_model.pt')
    EPOCHS = 50

    for epoch in range(EPOCHS):
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
        with torch.no_grad():
            for b_seq, b_feat, b_y in val_loader:
                logits = model(b_seq, b_feat)
                loss = loss_fn(logits, b_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            # print(f"    [Hybrid] Early stop at epoch {epoch + 1}")
            break

    model.load_state_dict(torch.load('temp_best_model.pt'))
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

    if os.path.exists('temp_best_model.pt'):
        os.remove('temp_best_model.pt')

    return weight_proba, test_proba, test_class


# ==========================================
# 工具函数 (修改重点)
# ==========================================
def smart_load_matrix(path):
    """
    智能读取 .txt 或 .csv 文件，增加了对混合类型的强力清洗
    """
    path = must_exist(path)
    try:
        # 1. 读取数据 (使用 low_memory=False 防止 DtypeWarning 警告不完整)
        # 优先尝试 CSV 格式
        if path.endswith('.csv'):
            try:
                # 尝试 header=None 读取
                df = pd.read_csv(path, header=None, low_memory=False)
            except:
                # 如果失败，可能是分隔符问题，尝试自动探测
                df = pd.read_csv(path, sep=None, engine='python', header=None)
        else:
            # 尝试空格分隔
            try:
                df = pd.read_csv(path, sep=r'\s+', header=None, low_memory=False)
            except:
                # 最后尝试逗号
                df = pd.read_csv(path, sep=',', header=None, low_memory=False)

        # 2. 强力清洗数据
        # 问题原因：某些列被读成了 object (string)，可能是因为第一行是表头

        # 尝试将所有列转换为数字，无法转换的变为 NaN (errors='coerce')
        df_numeric = df.apply(pd.to_numeric, errors='coerce')

        # 检查第一行：如果第一行包含大量 NaN，而后面几行正常，说明第一行是表头
        if df_numeric.iloc[0].isna().sum() > df.shape[1] * 0.5:
            # print(f"      [Info] Detected header in {os.path.basename(path)}, removing first row.")
            df_numeric = df_numeric.iloc[1:]

        # 3. 填充或删除剩余的 NaN
        # 如果某些数据本身就是脏的，填 0
        df_numeric.fillna(0.0, inplace=True)

        # 4. 转 Tensor
        data = df_numeric.values.astype(np.float32)
        return torch.tensor(data, dtype=torch.float32)

    except Exception as e:
        raise ValueError(f"无法读取文件 {path}, Error: {e}")


def load_data_for_cell_v2(cell_name, mode="train"):
    """
        修复版：显式指定文件名，防止读取到无关文件导致的数据泄露。
        """
    feature_dir = os.path.join(FEATURE_ROOT, cell_name, mode)
    data_dir = os.path.join(DATA_ROOT, cell_name, mode)

    print(f"  > Loading {cell_name} [{mode}]...")

    # 1. 加载标签
    y_filename = "y_train.txt" if mode == "train" else "y_test.txt"
    y_path = os.path.join(data_dir, y_filename)
    y = smart_load_matrix(y_path).view(-1)

    # ========================================================
    # 关键修改：显式定义文件名列表，确保顺序一致，且不读取杂乱文件
    # ========================================================

    # 手工特征文件名 (必须与单细胞脚本完全一致)
    HANDCRAFTED_FILES = [
        "cksnap.csv",
        "mismatch.csv",
        "rckmer.csv",
        "psetnc.csv",
        "tpcp.csv"
    ]

    # 序列特征文件名 (顺序很重要：先word2vec还是先dnabert必须固定)
    SEQ_FILES = [
        "word2vec.txt",  # 假设这是文件名
        "dnabert_features.txt"  # 假设这是文件名
    ]

    handcrafted_features = []
    seq_features = []

    # 2. 加载 Handcrafted Features
    for fname in HANDCRAFTED_FILES:
        fpath = os.path.join(feature_dir, fname)
        # 尝试加载 CSV 或 TXT
        if not os.path.exists(fpath):
            # 兼容可能的后缀差异，比如有的叫 .txt
            fpath_txt = fpath.replace('.csv', '.txt')
            if os.path.exists(fpath_txt):
                fpath = fpath_txt
            else:
                raise FileNotFoundError(f"Missing feature file: {fpath}")

        mat = smart_load_matrix(fpath)
        handcrafted_features.append(mat)

    # 3. 加载 Sequence Features
    for fname in SEQ_FILES:
        fpath = os.path.join(feature_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing sequence file: {fpath}")

        mat = smart_load_matrix(fpath)

        # 针对 word2vec/bert 可能的维度对齐问题（如果需要的话，可以在这里做padding逻辑）
        # 这里暂时假设 smart_load_matrix 已经处理好了或者文件本来就是对齐的
        seq_features.append(mat)

    # 4. 拼接
    x_feat = torch.cat(handcrafted_features, dim=1)
    x_seq = torch.cat(seq_features, dim=1)

    # 5. 维度对齐与截断 (防止 y 和 x 长度不一致)
    min_len = min(len(y), len(x_seq), len(x_feat))

    if len(y) != min_len:
        print(f"    [Warning] Length mismatch in {cell_name}! Truncating to {min_len}")

    y = y[:min_len]
    x_seq = x_seq[:min_len]
    x_feat = x_feat[:min_len]

    return x_feat, x_seq, y


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


def save_result_realtime(result_dict, filename):
    df = pd.DataFrame([result_dict])
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(filename, index=False, mode='a', header=False, encoding='utf-8-sig')


# ==========================================
# Main Execution
# ==========================================
def main():
    set_seed(2025)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    print(f"Results will be saved to: {OUTPUT_FILE}")

    # 外层循环：作为训练集的细胞系
    for train_cell in CELL_LINES:
        print(f"\n##################################################")
        print(f" STARTING TRAIN GROUP: {train_cell}")
        print(f"##################################################")

        try:
            x_train_rf_raw, x_train_seq, y_train = load_data_for_cell_v2(train_cell, mode="train")
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
                x_test_rf_raw, x_test_seq, y_test = load_data_for_cell_v2(test_cell, mode="test")
            except Exception as e:
                print(f"  [Error] Failed to load test data for {test_cell}: {e}")
                continue

            x_test_rf_norm = (x_test_rf_raw - min_v) / (range_v + 1e-8)

            # --- Model 1: Random Forest ---
            print("  Running RF...")
            rf_weight_proba, rf_test_proba, rf_test_class = RF.pred(
                x_rf_tr, y_tr, x_rf_w, x_test_rf_norm, n_trees=300
            )
            rf_auc, rf_acc, rf_mcc = calculate_metrics(y_test, rf_test_class, rf_test_proba)

            save_result_realtime({
                "Train Cell": train_cell, "Test Cell": test_cell, "Model": "RF",
                "AUC": rf_auc, "ACC": rf_acc, "MCC": rf_mcc
            }, OUTPUT_FILE)

            # --- Model 2: Hybrid Transformer ---
            print("  Running Hybrid-Trans...")
            cnn_weight_proba, cnn_test_proba, cnn_test_class = run_hybrid_training(
                X_seq_tr=x_seq_tr, X_feat_tr=x_rf_tr, y_tr=y_tr,
                X_seq_w=x_seq_w, X_feat_w=x_rf_w, y_w=y_w,
                X_seq_te=x_test_seq, X_feat_te=x_test_rf_norm
            )

            ht_auc, ht_acc, ht_mcc = calculate_metrics(y_test, cnn_test_class, cnn_test_proba)

            save_result_realtime({
                "Train Cell": train_cell, "Test Cell": test_cell, "Model": "Hybrid-trans",
                "AUC": ht_auc, "ACC": ht_acc, "MCC": ht_mcc
            }, OUTPUT_FILE)

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
                "Train Cell": train_cell, "Test Cell": test_cell, "Model": "Fusion",
                "AUC": fu_auc, "ACC": fu_acc, "MCC": fu_mcc
            }, OUTPUT_FILE)

    print(f"\nAll Done. File saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()