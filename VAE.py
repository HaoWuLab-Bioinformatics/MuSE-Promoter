import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
def set_seed(seed: int = 10):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_seq(x, embed_dim: int = 8):
    """
    输入处理：将 (N, D) 重塑为 (N, L, C)
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.dim() == 2:
        N, D = x.shape
        if embed_dim is not None and embed_dim > 0 and (D % embed_dim == 0) and (D // embed_dim) >= 2:
            x = x.view(N, D // embed_dim, embed_dim)
        else:
            x = x.unsqueeze(-1)
    return x


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len):
        super(VAE, self).__init__()

        # 输入的维度是 (batch_size, seq_len, input_dim)，seq_len 是序列长度，input_dim 是每个时间步的特征数

        # 编码器 (Encoder)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)  # 使用 LSTM 处理序列数据
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值 (mean)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 对数方差 (log-variance)

        # 解码器 (Decoder)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, input_dim, batch_first=True)  # 使用 LSTM 进行序列生成
        self.fc4 = nn.Linear(input_dim, input_dim)

    def encode(self, x):
        # 输入： (batch_size, seq_len, input_dim)
        _, (hn, _) = self.lstm(x)  # 只取最后一个时刻的隐藏状态
        h1 = hn[-1]  # 取 LSTM 的最后一个隐藏状态（可以尝试其他策略）
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # 重参数化技巧，采样潜在变量
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 标准正态分布的噪声
        return mu + eps * std

    def decode(self, z, seq_len):
        # 将潜在变量传递给解码器，生成序列数据
        h3 = F.relu(self.fc3(z))
        h3 = h3.unsqueeze(1).repeat(1, seq_len, 1)  # 将潜在变量扩展为与序列长度相同的形状
        out, _ = self.lstm2(h3)  # 使用 LSTM 解码序列
        return torch.sigmoid(self.fc4(out))  # 激活函数为 sigmoid，生成 [0, 1] 之间的概率

    def forward(self, x):
        mu, logvar = self.encode(x)  # 编码得到均值和对数方差
        z = self.reparameterize(mu, logvar)  # 通过重参数化技巧得到潜在变量
        return self.decode(z, x.size(1)), mu, logvar  # 返回重构的序列数据


def loss_function(recon_x, x, mu, logvar):
    """
    VAE 损失函数：
    - BCE（Binary Cross-Entropy）：衡量重构的序列与原始序列的相似度
    - KL散度：衡量潜在空间分布与标准正态分布的差异
    """
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  # 使用 sum 来计算每个元素的 BCE 损失

    # KL 散度（Kullback-Leibler divergence）
    # 使潜在变量的分布接近标准正态分布
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # 计算 KL 散度

    return BCE + KL


def _split_stratified(y, val_ratio=1 / 7, seed=10):
    """
    纯 torch 分层划分 indices
    y: (N,1) 或 (N,)
    """
    g = torch.Generator().manual_seed(seed)
    y = y.view(-1).to(torch.int64)

    idx0 = torch.nonzero(y == 0, as_tuple=False).view(-1)
    idx1 = torch.nonzero(y == 1, as_tuple=False).view(-1)

    if idx0.numel() == 0 or idx1.numel() == 0:
        # 没法分层，直接随机切
        perm = torch.randperm(y.numel(), generator=g)
        n_val = max(1, int(y.numel() * val_ratio))
        return perm[n_val:], perm[:n_val]

    idx0 = idx0[torch.randperm(idx0.numel(), generator=g)]
    idx1 = idx1[torch.randperm(idx1.numel(), generator=g)]

    n0_val = max(1, int(idx0.numel() * val_ratio))
    n1_val = max(1, int(idx1.numel() * val_ratio))

    val_idx = torch.cat([idx0[:n0_val], idx1[:n1_val]], dim=0)
    tr_idx = torch.cat([idx0[n0_val:], idx1[n1_val:]], dim=0)

    val_idx = val_idx[torch.randperm(val_idx.numel(), generator=g)]
    tr_idx = tr_idx[torch.randperm(tr_idx.numel(), generator=g)]
    return tr_idx, val_idx


def pred(x_train, y_train, x_weight, x_test, learning_rate, KERNEL_NUM, KERNEL_SIZE):
    """
    为了兼容你原 main：参数名保留
    KERNEL_NUM / KERNEL_SIZE 映射成 VAE 超参（合理即可）

    返回:
      weight_proba: (Nw,1)
      test_proba:   (Nt,1)
      test_class:   (Nt,1)
    """
    set_seed(10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 关键修复：这里会把 (N,1258) 自动变成 (N, L, C) =====
    X  = _as_seq(x_train, embed_dim=8)
    y  = torch.as_tensor(y_train, dtype=torch.float32).view(-1, 1)

    Xw = _as_seq(x_weight, embed_dim=8)
    Xt = _as_seq(x_test, embed_dim=8)

    # 自动计算序列长度
    seq_len = X.shape[1]  # 假设输入数据的形状是 (batch_size, seq_len, input_dim)

    # 映射超参
    latent_dim = int(max(64, min(256, KERNEL_NUM * 4)))
    hidden_dim = int(latent_dim * 2)

    in_channels = int(X.shape[-1])  # 若输入是 1258，会变成 8
    model = VAE(
        input_dim=in_channels,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        seq_len=seq_len  # 传入动态计算的 seq_len
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-2)

    tr_idx, val_idx = _split_stratified(y, val_ratio=1/7, seed=10)
    Xtr, ytr = X[tr_idx].to(device), y[tr_idx].to(device)
    Xva, yva = X[val_idx].to(device), y[val_idx].to(device)

    max_epoch = 100
    batch_size = 50
    patience = 5
    best_val = float("inf")
    bad = 0
    best_state = None

    for epoch in range(1, max_epoch + 1):
        model.train()
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], batch_size):
            b = perm[i:i + batch_size]
            xb = Xtr[b]
            yb = ytr[b]
            opt.zero_grad()
            recon_batch, mu, logvar = model(xb)
            loss = loss_function(recon_batch, xb, mu, logvar)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_function(model(Xva), Xva, mu, logvar).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict_proba(Xinp):
        Xinp = Xinp.to(device)
        with torch.no_grad():
            recon_batch, _, _ = model(Xinp)
        return recon_batch.view(-1, 1)

    weight_proba = predict_proba(Xw)
    test_proba = predict_proba(Xt)
    test_class = (test_proba >= 0.5).to(torch.int64)

    return weight_proba, test_proba, test_class
