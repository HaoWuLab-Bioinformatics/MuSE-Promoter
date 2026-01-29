# RF.py
import os
import random
import torch
import torch.nn as nn

def set_seed(seed: int = 10):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # logits
        )

    def forward(self, x):
        return self.net(x)

def _to_float_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)

def _to_label_tensor(y):
    y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
    return y

def pred(x_train, y_train, x_weight, x_test, n_trees=300,
         lr=1e-3, max_epoch=200, batch_size=128, patience=10, seed=10):
    """
    返回:
      weight_proba: (Nw,1)
      test_proba:   (Nt,1)
      test_class:   (Nt,1)
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = _to_float_tensor(x_train)
    Ytr = _to_label_tensor(y_train)
    Xw  = _to_float_tensor(x_weight)
    Xt  = _to_float_tensor(x_test)

    in_dim = Xtr.shape[1]
    model = MLPClassifier(in_dim).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    # 简单 early stopping：用 x_weight / y_weight 做验证更合理，但你原逻辑里 weight 集是给融合用的
    # 这里我们用训练集再切 1/7 做 val（纯 torch 实现）
    N = Xtr.shape[0]
    perm = torch.randperm(N)
    val_n = max(1, N // 7)
    val_idx = perm[:val_n]
    tr_idx  = perm[val_n:]

    X_train = Xtr[tr_idx].to(device)
    y_train = Ytr[tr_idx].to(device)
    X_val   = Xtr[val_idx].to(device)
    y_val   = Ytr[val_idx].to(device)

    best_val = float("inf")
    bad = 0
    best_state = None

    for epoch in range(1, max_epoch + 1):
        model.train()
        # minibatch
        idx = torch.randperm(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            b = idx[i:i+batch_size]
            xb = X_train[b]
            yb = y_train[b]

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # val
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val).item()

        # early stop
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

    def predict_proba(X):
        X = X.to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(X)).cpu()
        return p

    weight_proba = predict_proba(Xw).view(-1, 1)
    test_proba = predict_proba(Xt).view(-1, 1)
    test_class = (test_proba >= 0.5).to(torch.int64)

    return weight_proba, test_proba, test_class
import torch
import torch.nn as nn


def _to_1d_float(x):
    """兼容 torch / numpy / list -> torch.float32 1D"""
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    return t.to(torch.float32).view(-1)


def _bce(y, p, eps=1e-6):
    """对概率做 BCE（log-loss）"""
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p)).mean()


def _fuse_prob(w1, p1, p2):
    """p = w1*p1 + (1-w1)*p2"""
    return w1 * p1 + (1.0 - w1) * p2


def weight(y_weight, rf_weight_proba, cnn_weight_proba, rf_test_proba, cnn_test_proba,
           iterations_num=50, steps=300, lr=0.1, seed=10, verbose=True):
    """
    PyTorch 版两模型加权融合（替代 scipy.minimize + sklearn.log_loss）

    输入:
      y_weight:         (Nw,) or (Nw,1)
      rf_weight_proba:  (Nw,) or (Nw,1)
      cnn_weight_proba: (Nw,) or (Nw,1)
      rf_test_proba:    (Nt,) or (Nt,1)
      cnn_test_proba:   (Nt,) or (Nt,1)

    输出:
      final_prediction:       (Nt,1) 概率
      final_prediction_label: (Nt,1) 0/1
    """
    # 固定随机种子（只影响初始化）
    g = torch.Generator().manual_seed(seed)

    y = _to_1d_float(y_weight)
    p1w = _to_1d_float(rf_weight_proba)
    p2w = _to_1d_float(cnn_weight_proba)

    p1t = _to_1d_float(rf_test_proba)
    p2t = _to_1d_float(cnn_test_proba)

    # 记录最优
    best_loss = float("inf")
    best_w1 = 0.5

    for it in range(iterations_num):
        # 用一个自由参数 a -> w1=sigmoid(a) 保证 w1 in (0,1)
        # 随机初始化 a，相当于你原来随机初始化 weights
        a = torch.randn((), generator=g, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([a], lr=lr)

        for _ in range(steps):
            w1 = torch.sigmoid(a)
            pw = _fuse_prob(w1, p1w, p2w)
            loss = _bce(y, pw)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # 评估这次初始化的最优
        with torch.no_grad():
            w1 = torch.sigmoid(a).item()
            pw = _fuse_prob(torch.tensor(w1), p1w, p2w)
            loss_val = _bce(y, pw).item()

        if loss_val < best_loss:
            best_loss = loss_val
            best_w1 = w1

    # 用最优 w1 做 test 融合
    w1 = torch.tensor(best_w1, dtype=torch.float32)
    final_prediction = _fuse_prob(w1, p1t, p2t).view(-1, 1)
    final_prediction_label = (final_prediction >= 0.5).to(torch.int64)

    if verbose:
        print(f"[Fusion] best_weights=[{best_w1:.6f}, {1.0-best_w1:.6f}] best_logloss={best_loss:.6f}")

    return final_prediction, final_prediction_label
