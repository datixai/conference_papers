"""Window risk models. Neural head does not see port features; the lexical stub does."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest


STRUCT = ["n_edge", "n_src", "n_dst", "max_out", "cross", "dur", "fwd"]
PORT = ["n_port", "high_port"]


def matrix(feat: dict, keys: list[str]) -> np.ndarray:
    x = np.stack([np.log1p(np.abs(feat[k].astype(np.float64))) if k != "cross" else feat[k].astype(np.float64) for k in keys], axis=1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


def temporal_ema(max_out: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    ema = np.zeros_like(max_out, dtype=np.float64)
    v = 0.0
    for i, m in enumerate(max_out):
        v = alpha * v + (1.0 - alpha) * float(m)
        ema[i] = v
    return ema


def add_temporal(feat: dict) -> dict:
    feat = dict(feat)
    feat["ema_out"] = temporal_ema(feat["max_out"])
    return feat


class WindowMLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def standardize(x_train: np.ndarray, x_all: np.ndarray) -> np.ndarray:
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0) + 1e-6
    return (x_all - mu) / sd


def train_mlp(x: np.ndarray, y: np.ndarray, seed: int, epochs: int = 40, lr: float = 2e-3) -> WindowMLP:
    torch.manual_seed(seed)
    model = WindowMLP(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.as_tensor(x)
    yt = torch.as_tensor(y, dtype=torch.float32)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    pw = torch.tensor(neg / pos)
    model.train()
    for _ in range(epochs):
        logit = model(xt)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, yt, pos_weight=pw)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def mlp_risk(model: WindowMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    logit = model(torch.as_tensor(x))
    return torch.sigmoid(logit).cpu().numpy()


def iforest_risk(x_train: np.ndarray, x_all: np.ndarray, seed: int) -> np.ndarray:
    clf = IsolationForest(n_estimators=80, contamination=0.2, random_state=seed, n_jobs=1)
    clf.fit(x_train)
    s_tr, s = -clf.score_samples(x_train), -clf.score_samples(x_all)
    lo, hi = np.percentile(s_tr, 1), np.percentile(s_tr, 99)  # train-only scaling
    return np.clip((s - lo) / max(hi - lo, 1e-9), 0, 1)


def degree_risk(max_out: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    thr = np.quantile(max_out[train_idx], 0.70)
    r = max_out / max(thr, 1.0)
    return np.clip(r, 0, 1)


def lexical_stub(feat: dict, train_idx: np.ndarray) -> np.ndarray:
    port_thr = np.quantile(feat["n_port"][train_idx], 0.75)
    hp_thr = np.quantile(feat["high_port"][train_idx], 0.70)
    return ((feat["n_port"] >= port_thr) | (feat["high_port"] >= hp_thr)).astype(np.int8)
