import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 10):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PositionalEncoding(nn.Module):
    """
    动态位置编码：支持任意长度序列输入
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        # 预先生成一个足够长的 PE
        pe = self._build_pe(max_len, d_model)
        self.register_buffer("pe", pe)

    def _build_pe(self, length, d_model):
        pe = torch.zeros(1, length, d_model)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # x: (B, L, D)
        L = x.size(1)
        # 如果当前序列长度超过缓存，重新生成
        if L > self.pe.size(1):
            self.pe = self._build_pe(L, self.d_model).to(x.device)
        return x + self.pe[:, :L, :]


class ConvTokenizer(nn.Module):
    """
    将长序列 (N, 1, 8000) 压缩并提取特征 -> (N, d_model, L_new)
    类似于 Vision Transformer 的 Patch Embedding，但这里使用多层 Conv1d
    """

    def __init__(self, in_channels=1, d_model=128):
        super().__init__()
        # Layer 1: 8000 -> 2000
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        # Layer 2: 2000 -> 500
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        # Layer 3: 500 -> 250 (映射到 d_model)
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # x: (N, 1, L_raw)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x  # (N, d_model, L_reduced)


class Transformer1DClassifier(nn.Module):
    def __init__(self, input_len=8000, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        # 1. 特征提取与降维 (CNN Tokenizer)
        self.tokenizer = ConvTokenizer(in_channels=1, d_model=d_model)

        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (N, 8000) 或者是 (N, 8000, 1)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (N, 1, 8000)
        elif x.dim() == 3 and x.shape[-1] == 1:
            x = x.permute(0, 2, 1)  # (N, 1, 8000)

        # CNN 提取特征
        x = self.tokenizer(x)  # (N, d_model, L')

        # Transformer 需要 (N, L', d_model)
        x = x.permute(0, 2, 1)  # (N, L', d_model)

        # 加上位置编码
        x = self.pos_encoder(x)

        # Transformer 编码
        x = self.transformer(x)  # (N, L', d_model)

        # Global Average Pooling
        x = x.mean(dim=1)  # (N, d_model)

        # 分类
        logits = self.head(x)
        return logits


def _split_stratified(y, val_ratio=1 / 7, seed=10):
    g = torch.Generator().manual_seed(seed)
    y = y.view(-1).long()

    idx0 = torch.nonzero(y == 0, as_tuple=False).view(-1)
    idx1 = torch.nonzero(y == 1, as_tuple=False).view(-1)

    # Shuffle
    idx0 = idx0[torch.randperm(idx0.numel(), generator=g)]
    idx1 = idx1[torch.randperm(idx1.numel(), generator=g)]

    n0_val = max(1, int(idx0.numel() * val_ratio))
    n1_val = max(1, int(idx1.numel() * val_ratio))

    val_idx = torch.cat([idx0[:n0_val], idx1[:n1_val]], dim=0)
    tr_idx = torch.cat([idx0[n0_val:], idx1[n1_val:]], dim=0)

    # Shuffle again to mix 0 and 1
    val_idx = val_idx[torch.randperm(val_idx.numel(), generator=g)]
    tr_idx = tr_idx[torch.randperm(tr_idx.numel(), generator=g)]
    return tr_idx, val_idx


def pred(x_train, y_train, x_weight, x_test, learning_rate, KERNEL_NUM, KERNEL_SIZE):
    """
    接口保持不变，但内部实现升级为 CNN-Transformer 混合架构。
    KERNEL_NUM / KERNEL_SIZE 参数用于微调 d_model 和 layers。
    """
    set_seed(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转换为 Tensor，保持原始形状 (N, 8000)
    # 不再进行奇怪的 reshape，交给模型内部的 CNN 处理
    X = torch.as_tensor(x_train, dtype=torch.float32)
    y = torch.as_tensor(y_train, dtype=torch.float32).view(-1, 1)
    Xw = torch.as_tensor(x_weight, dtype=torch.float32)
    Xt = torch.as_tensor(x_test, dtype=torch.float32)

    # 映射超参数 (让传入的参数有意义)
    # d_model 设为 128 或 256
    d_model = 128 if KERNEL_NUM < 64 else 256
    # num_layers 设为 2 或 3
    num_layers = 2 if KERNEL_SIZE < 4 else 3

    # 初始化模型
    model = Transformer1DClassifier(
        input_len=X.shape[1],  # 8000
        d_model=d_model,
        nhead=4,
        num_layers=num_layers,
        dropout=0.3
    ).to(device)

    # 使用 AdamW，稍微增大权重衰减防止过拟合
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=False)

    # 划分训练/验证集
    tr_idx, val_idx = _split_stratified(y, val_ratio=0.15, seed=2025)
    Xtr, ytr = X[tr_idx].to(device), y[tr_idx].to(device)
    Xva, yva = X[val_idx].to(device), y[val_idx].to(device)

    max_epoch = 200  # 收敛会更快，不需要太久
    batch_size = 64
    patience = 10
    best_val = float("inf")
    bad = 0
    best_state = None

    for epoch in range(1, max_epoch + 1):
        model.train()
        perm = torch.randperm(Xtr.shape[0])

        epoch_loss = 0
        steps = 0

        for i in range(0, Xtr.shape[0], batch_size):
            b_idx = perm[i:i + batch_size]
            xb = Xtr[b_idx]
            yb = ytr[b_idx]

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()

            # 梯度裁剪，防止 Transformer 训练不稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            steps += 1

        # 验证
        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_loss = loss_fn(val_logits, yva).item()

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                # print(f"Early stopping at epoch {epoch}")
                break

    # 加载最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    def predict_proba(Xinp):
        # 这里的 Xinp 已经是 Tensor，只需转设备
        dataset = torch.utils.data.TensorDataset(Xinp)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                xb = batch[0].to(device)
                p = torch.sigmoid(model(xb))
                preds.append(p.cpu())
        return torch.cat(preds, dim=0).view(-1, 1)

    weight_proba = predict_proba(Xw)
    test_proba = predict_proba(Xt)
    test_class = (test_proba >= 0.5).long()

    return weight_proba, test_proba, test_class