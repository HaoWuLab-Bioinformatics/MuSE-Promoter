# TRANSFORMER.py
import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 10):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NumpySeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray = None):
        """
        x: (N, L) or (N, L, C)
        y: (N,) or None
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = np.expand_dims(x, axis=-1)  # (N, L, 1)
        self.x = torch.from_numpy(x)

        if y is None:
            self.y = None
        else:
            y = np.asarray(y, dtype=np.float32).reshape(-1, 1)  # (N,1)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        L = x.size(1)
        return x + self.pe[:, :L, :]


class Transformer1DClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
        pooling: str = "mean",   # "mean" / "max" / "cls"
        max_len: int = 5000
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.pooling = pooling

        self.input_proj = nn.Linear(in_channels, d_model)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # logits
        )

        nn.init.normal_(self.input_proj.weight, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, L, C)
        x = self.input_proj(x)  # (B, L, D)

        if self.pooling == "cls":
            B = x.size(0)
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
            x = torch.cat([cls, x], dim=1)          # (B,1+L,D)

        x = self.pos(x)
        x = self.drop(x)
        x = self.encoder(x)

        if self.pooling == "mean":
            feat = x.mean(dim=1)
        elif self.pooling == "max":
            feat = x.max(dim=1).values
        else:  # cls
            feat = x[:, 0, :]

        logits = self.head(feat)  # (B,1)
        return logits


def _train_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float,
    max_epoch: int,
    batch_size: int,
    patience: int,
    model_path: str,
    d_model: int,
    num_heads: int,
    num_layers: int,
    ff_dim: int,
    dropout: float,
    pooling: str,
    seed: int = 10,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=1/7, random_state=10, stratify=y_train
    )

    tr_ds = NumpySeqDataset(x_tr, y_tr)
    va_ds = NumpySeqDataset(x_val, y_val)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # infer channels
    sample_x, _ = tr_ds[0]  # (L,C)
    in_channels = sample_x.shape[-1]

    model = Transformer1DClassifier(
        in_channels=in_channels,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        pooling=pooling,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    bad = 0

    for epoch in range(1, max_epoch + 1):
        model.train()
        tr_losses = []
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_losses.append(loss.item())

        tr_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_loss = float(np.mean(va_losses)) if va_losses else 0.0
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            bad = 0
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model: {model_path}")
        else:
            bad += 1
            if bad >= patience:
                print(f"  ✋ Early stopping (patience={patience})")
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


@torch.no_grad()
def _predict(model, device, x: np.ndarray, batch_size: int = 256):
    ds = NumpySeqDataset(x, y=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    probs = []
    for xb in loader:
        xb = xb.to(device)
        logits = model(xb)                # (B,1)
        p = torch.sigmoid(logits)         # (B,1)
        probs.append(p.cpu().numpy())

    proba = np.vstack(probs).astype(np.float64)     # (N,1)
    pred_class = (proba >= 0.5).astype(np.int32)    # (N,1)
    return proba, pred_class


# ----------------------------
# 对齐你原 CNN.pred 的接口：
# pred(x_train, y_train, x_weight, x_test, learning_rate, KERNEL_NUM, KERNEL_SIZE)
# 这里把 KERNEL_NUM/KERNEL_SIZE 映射到 Transformer 的超参（不影响你主程序调用）
# ----------------------------
def pred(x_train, y_train, x_weight, x_test, learning_rate, KERNEL_NUM, KERNEL_SIZE):
    """
    返回：
      weight_proba: (Nw,1)
      test_proba: (Nt,1)
      test_class: (Nt,1)
    """
    # 让你原来的参数仍然能用：
    # - KERNEL_NUM -> d_model（常见设为 64/128/256）
    # - KERNEL_SIZE -> ff_dim 的缩放（随便映射一个合理值）
    d_model = int(max(64, min(256, KERNEL_NUM * 4)))
    ff_dim = int(max(128, d_model * 2))
    num_heads = 4 if d_model >= 128 else 2
    num_layers = 3
    dropout = 0.1
    pooling = "mean"

    model, device = _train_one(
        x_train=np.asarray(x_train),
        y_train=np.asarray(y_train),
        learning_rate=float(learning_rate),
        max_epoch=100,
        batch_size=50,
        patience=5,
        model_path="model_transformer.pt",
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        pooling=pooling,
        seed=10,
    )

    weight_proba, _ = _predict(model, device, np.asarray(x_weight))
    test_proba, test_class = _predict(model, device, np.asarray(x_test))

    return weight_proba, test_proba, test_class
