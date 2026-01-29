import os
# 【新增】必须放在最开头，解决 PyTorch 和 t-SNE 的线程冲突死锁
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import csv
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score

# ==========================================
# 假设这些模块你本地有，保持引用
# 请确保当前目录下有 RF.py 和 Weighted_average_trans.py
import Weighted_average_trans
import RF

# ==========================================

# ================= 配置区域 =================
# 根目录 (请务必修改为你的实际路径)
PROJECT_ROOT = "/home/user012/experments/Desktop/pythonProjectexperments/iPro-WAEL-main/iPro-WAEL-main/EPfeature"

# 定义所有需要验证的细胞系
CELL_LINES = ["GM12878", "HeLa-S3", "HUVEC", "K562"]

# 结果保存文件名
REALTIME_FILE = "Cross_Cell_Validation_Results.csv"
TSNE_DIR = "TSNE_Results"


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
# 模型架构
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
                pass  # 减少日志输出
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

    def forward(self, x_seq, x_feat, return_embedding=False):
        """
        新增 return_embedding 参数，用于 t-SNE 可视化
        """
        x = x_seq.unsqueeze(-1)
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        x = self.cnn_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        seq_emb = x.mean(dim=1)
        feat_emb = self.feat_mlp(x_feat)

        # 融合特征
        combined = torch.cat([seq_emb, feat_emb], dim=1)

        logits = self.classifier(combined)

        if return_embedding:
            return logits, combined
        return logits


# ==========================================
# 训练与预测流程
# ==========================================
def run_hybrid_training(X_seq_tr, X_feat_tr, y_tr, X_seq_w, X_feat_w, y_w, X_seq_te, X_feat_te):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据转换
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

    early_stopping = EarlyStopping(patience=5, verbose=True, path='temp_best_model.pt')
    EPOCHS = 50

    # --- Training Loop ---
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
            print(f"  [Info] Early stopping at epoch {epoch + 1}")
            break

    # --- Prediction & Feature Extraction ---
    model.load_state_dict(torch.load('temp_best_model.pt'))
    model.eval()

    def predict_and_extract(x_s, x_f):
        ds = TensorDataset(torch.as_tensor(x_s, dtype=torch.float32), torch.as_tensor(x_f, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=128, shuffle=False)
        probs = []
        embeddings = []
        with torch.no_grad():
            for b_s, b_f in dl:
                b_s, b_f = b_s.to(device), b_f.to(device)
                logit, emb = model(b_s, b_f, return_embedding=True)  # 获取特征
                p = torch.sigmoid(logit)
                probs.append(p.cpu())
                embeddings.append(emb.cpu())
        return torch.cat(probs, dim=0).view(-1, 1), torch.cat(embeddings, dim=0)

    weight_proba, _ = predict_and_extract(X_seq_w, X_feat_w)
    test_proba, test_embeddings = predict_and_extract(X_seq_te, X_feat_te)
    test_class = (test_proba >= 0.5).long()

    if os.path.exists('temp_best_model.pt'):
        os.remove('temp_best_model.pt')

    return weight_proba, test_proba, test_class, test_embeddings


# ==========================================
# 可视化工具
# ==========================================
def plot_tsne(features, labels, title, save_path):
    """
    绘制 t-SNE 散点图 (优化版)
    """
    print(f"  [Visual] Generating t-SNE for {title}...")

    # 颜色风格设置
    sns.set_style("whitegrid")

    # --- 优化修改点 ---
    # 1. verbose=1: 显示进度条，不再“假死”
    # 2. n_iter=300: 减少迭代次数，大幅加速
    # 3. method='barnes_hut': 确保使用加速算法
    tsne = TSNE(n_components=2, perplexity=30, random_state=2025,
                n_iter=300, verbose=1, method='barnes_hut',
                init='pca', learning_rate='auto')

    # 开始降维
    try:
        X_embedded = tsne.fit_transform(features)
    except Exception as e:
        print(f"  [Skipped] t-SNE failed with error: {e}")
        return

    plt.figure(figsize=(10, 8))

    # 绘制两类点
    # 0: Negative, 1: Positive
    indices_0 = (labels == 0)
    indices_1 = (labels == 1)

    plt.scatter(X_embedded[indices_0, 0], X_embedded[indices_0, 1],
                c='#3498db', label='Non-Enhancer', alpha=0.6, s=15, edgecolors='w', linewidth=0.5)
    plt.scatter(X_embedded[indices_1, 0], X_embedded[indices_1, 1],
                c='#e74c3c', label='Enhancer', alpha=0.6, s=15, edgecolors='w', linewidth=0.5)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', frameon=True)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [Visual] Saved image to {save_path}")


# ==========================================
# 数据加载与处理工具
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
    raw_data = {}
    raw_data['y'] = load_txt_vector(os.path.join(base_dir, "y.txt"))
    features = ["cksnap", "mismatch", "rckmer", "psetnc", "tpcp"]
    for i, feature in enumerate(features):
        raw_data[f'rf_{i}'] = load_whitespace_matrix(os.path.join(base_dir, f"{feature}.txt"))
    raw_data['w2v'] = load_whitespace_matrix(os.path.join(base_dir, "word2vec.txt"))
    bert_path = os.path.join(base_dir, "dnabert_features.txt")
    if os.path.exists(bert_path):
        raw_data['bert'] = load_whitespace_matrix(bert_path)
    else:
        raw_data['bert'] = None

    aligned = align_and_truncate(raw_data)
    y = aligned['y']
    rf_list = [aligned[f'rf_{i}'] for i in range(len(features))]
    x_rf = torch.cat(rf_list, dim=1)
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

    # 确保保存 t-SNE 图片的文件夹存在
    if not os.path.exists(TSNE_DIR):
        os.makedirs(TSNE_DIR)

    print(f"Results will be saved to: {REALTIME_FILE}")
    print(f"t-SNE plots will be saved to: {TSNE_DIR}/")

    for train_cell in CELL_LINES:
        print(f"\n##################################################")
        print(f" STARTING TRAIN GROUP: {train_cell}")
        print(f"##################################################")

        # 1. Load Train Data
        try:
            x_train_rf_raw, x_train_seq, y_train = load_data_for_cell(train_cell, "train")
        except Exception as e:
            print(f"Skipping {train_cell} due to error: {e}")
            continue

        # Normalize
        min_v = x_train_rf_raw.min(0).values
        max_v = x_train_rf_raw.max(0).values
        range_v = max_v - min_v
        range_v[range_v == 0] = 1.0
        x_train_rf_norm = (x_train_rf_raw - min_v) / range_v

        # Split
        (x_rf_tr, x_rf_w, x_seq_tr, x_seq_w, y_tr, y_w) = stratified_split_multi(
            x_train_rf_norm, x_train_seq, y_train, test_size=1 / 8, seed=10
        )

        for test_cell in CELL_LINES:
            print(f"\n>>> Case: Train[{train_cell}] vs Test[{test_cell}]")
            torch.cuda.empty_cache()

            # 2. Load Test Data
            try:
                x_test_rf_raw, x_test_seq, y_test = load_data_for_cell(test_cell, "test")
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
            }, REALTIME_FILE)

            # --- Model 2: Hybrid Transformer with t-SNE Extraction ---
            print("  Running Hybrid-Trans...")
            cnn_weight_proba, cnn_test_proba, cnn_test_class, cnn_test_embeddings = run_hybrid_training(
                X_seq_tr=x_seq_tr, X_feat_tr=x_rf_tr, y_tr=y_tr,
                X_seq_w=x_seq_w, X_feat_w=x_rf_w, y_w=y_w,
                X_seq_te=x_test_seq, X_feat_te=x_test_rf_norm
            )
            ht_auc, ht_acc, ht_mcc = calculate_metrics(y_test, cnn_test_class, cnn_test_proba)

            save_result_realtime({
                "Train Cell": train_cell, "Test Cell": test_cell, "Model": "Hybrid-trans",
                "AUC": ht_auc, "ACC": ht_acc, "MCC": ht_mcc
            }, REALTIME_FILE)

            # --- VISUALIZATION: t-SNE ---
            # 如果数据量太大，随机采样 2000 个点，避免画图太慢
            emb_np = cnn_test_embeddings.numpy()
            y_test_np = y_test.detach().cpu().numpy().flatten()

            MAX_SAMPLES = 1000
            plot_name = f"{train_cell}_to_{test_cell}.png"
            plot_path = os.path.join(TSNE_DIR, plot_name)
            plot_title = f"Features: Train {train_cell} -> Test {test_cell}"

            if len(y_test_np) > MAX_SAMPLES:
                # 保持随机种子一致以便复现
                np.random.seed(2025)
                indices = np.random.choice(len(y_test_np), MAX_SAMPLES, replace=False)
                plot_tsne(emb_np[indices], y_test_np[indices], plot_title, plot_path)
            else:
                plot_tsne(emb_np, y_test_np, plot_title, plot_path)

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
            }, REALTIME_FILE)

            print(f"  [Done] {train_cell}->{test_cell} finished.")

    print(f"\nAll validations completed. Check {REALTIME_FILE} and {TSNE_DIR} folder.")


if __name__ == "__main__":
    main()