import argparse
import json
import math
import os
import random
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Reproducibility
# -----------------------------
def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Download / dataset utilities
# -----------------------------
NAB_ZIP_URL = "https://github.com/numenta/NAB/archive/refs/heads/master.zip"


def download_and_extract_nab(root_dir: str) -> str:
    os.makedirs(root_dir, exist_ok=True)
    zip_path = os.path.join(root_dir, "NAB-master.zip")
    repo_root = os.path.join(root_dir, "NAB-master")

    if os.path.isdir(repo_root) and os.path.isdir(os.path.join(repo_root, "data")):
        return repo_root

    import urllib.request
    print(f"[download] Fetching: {NAB_ZIP_URL}")
    urllib.request.urlretrieve(NAB_ZIP_URL, zip_path)

    print(f"[download] Extracting to: {root_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root_dir)

    try:
        os.remove(zip_path)
    except OSError:
        pass

    if not os.path.isdir(repo_root):
        raise RuntimeError("Could not find extracted NAB-master directory.")
    return repo_root


def _safe_to_datetime(series: pd.Series) -> pd.DatetimeIndex:
    dt = pd.to_datetime(series, errors="coerce")
    if dt.isna().any():
        dt = pd.to_datetime(series.astype(str), errors="coerce")
    if dt.isna().any():
        raise ValueError("Failed to parse some timestamps.")
    return pd.DatetimeIndex(dt)


def load_nab_labels(labels_path: str) -> Dict[str, List[str]]:
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_files_balanced(labels: Dict[str, List[str]], data_dir: str, max_files: int) -> List[str]:
    preferred_groups = [
        "realKnownCause",
        "realAWSCloudwatch",
        "realAdExchange",
        "realTraffic",
        "realTweets",
        "artificialWithAnomaly",
        "artificialNoAnomaly",
    ]

    existing = []
    for k in labels.keys():
        fp = os.path.join(data_dir, k)
        if os.path.isfile(fp):
            existing.append(k)

    grouped: Dict[str, List[str]] = {g: [] for g in preferred_groups}
    others: List[str] = []

    for k in existing:
        group = k.split("/")[0]
        if group in grouped:
            grouped[group].append(k)
        else:
            others.append(k)

    for g in grouped:
        grouped[g] = sorted(grouped[g])

    selected: List[str] = []
    while len(selected) < max_files:
        progressed = False
        for g in preferred_groups:
            if len(selected) >= max_files:
                break
            if grouped[g]:
                selected.append(grouped[g].pop(0))
                progressed = True
        if not progressed:
            break

    if len(selected) < max_files:
        selected.extend(sorted(others)[: (max_files - len(selected))])

    return selected[:max_files]


@dataclass
class SeriesData:
    key: str
    df: pd.DataFrame
    train_end: int
    val_end: int


def load_one_series(data_dir: str, key: str, label_ts: List[str], label_radius: int = 0) -> SeriesData:
    path = os.path.join(data_dir, key)
    raw = pd.read_csv(path)
    if raw.shape[1] < 2:
        raise ValueError(f"{key} has < 2 columns; cannot parse.")

    ts = _safe_to_datetime(raw.iloc[:, 0])
    values = pd.to_numeric(raw.iloc[:, 1], errors="coerce").astype(float)
    if values.isna().any():
        values = values.interpolate(limit_direction="both")

    df = pd.DataFrame({"timestamp": ts, "value": values})

    # FIX: duplicates timestamps (AdExchange) -> average duplicates
    df = df.groupby("timestamp", as_index=False)["value"].mean()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["label"] = 0

    idx = pd.DatetimeIndex(df["timestamp"])
    if len(idx) >= 3:
        median_delta = (idx[1:] - idx[:-1]).median()
        if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
            median_delta = pd.Timedelta("1H")
    else:
        median_delta = pd.Timedelta("1H")
    tol = median_delta * 2

    for s in label_ts:
        lab_dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(lab_dt):
            continue
        pos = idx.get_indexer([lab_dt], method="nearest", tolerance=tol)
        if pos is None or pos.size == 0:
            continue
        p = int(pos[0])
        if p < 0:
            continue
        lo = max(0, p - label_radius)
        hi = min(len(df) - 1, p + label_radius)
        df.loc[lo:hi, "label"] = 1

    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    train_end = max(train_end, 10)
    val_end = max(val_end, train_end + 10)
    val_end = min(val_end, n - 1)

    return SeriesData(key=key, df=df, train_end=train_end, val_end=val_end)


def make_windows(values: np.ndarray, labels: np.ndarray, window: int, stride: int):
    X, y, pos = [], [], []
    n = len(values)
    for end in range(window - 1, n, stride):
        start = end - window + 1
        X.append(values[start:end + 1])
        y.append(labels[end])
        pos.append(end)
    X = np.asarray(X, dtype=np.float32).reshape(-1, window, 1)
    y = np.asarray(y, dtype=np.int64)
    pos = np.asarray(pos, dtype=np.int64)
    return X, y, pos


@dataclass
class SplitPack:
    X_train: np.ndarray
    y_train: np.ndarray
    meta_train: List[Dict[str, Any]]
    X_val: np.ndarray
    y_val: np.ndarray
    meta_val: List[Dict[str, Any]]
    X_test: np.ndarray
    y_test: np.ndarray
    meta_test: List[Dict[str, Any]]


def prepare_dataset(series_list: List[SeriesData], window: int, stride: int,
                    train_only_normal: bool, max_train_windows: int) -> SplitPack:
    Xtr, ytr, mtr = [], [], []
    Xva, yva, mva = [], [], []
    Xte, yte, mte = [], [], []

    for sd in series_list:
        df = sd.df.copy()

        scaler = StandardScaler()
        train_vals = df.loc[: sd.train_end - 1, ["value"]].values
        scaler.fit(train_vals)

        df["value_z"] = scaler.transform(df[["value"]].values).reshape(-1)

        train_df = df.iloc[: sd.train_end].reset_index(drop=True)
        val_df = df.iloc[sd.train_end: sd.val_end].reset_index(drop=True)
        test_df = df.iloc[sd.val_end:].reset_index(drop=True)

        def build(split_df: pd.DataFrame, split_name: str):
            vals = split_df["value_z"].values.astype(np.float32)
            labs = split_df["label"].values.astype(np.int64)
            if len(vals) < window:
                return None
            X, y, end_pos = make_windows(vals, labs, window=window, stride=stride)

            meta = []
            ts_arr = split_df["timestamp"].values
            raw_val = split_df["value"].values.astype(float)
            for p in end_pos:
                meta.append({
                    "series_key": sd.key,
                    "split": split_name,
                    "end_pos_in_split": int(p),
                    "timestamp": pd.Timestamp(ts_arr[p]).isoformat(),
                    "raw_value": float(raw_val[p]),
                })
            return X, y, meta

        out_tr = build(train_df, "train")
        out_va = build(val_df, "val")
        out_te = build(test_df, "test")

        if out_tr is not None:
            X, y, meta = out_tr
            if train_only_normal:
                mask = (y == 0)
                X, y = X[mask], y[mask]
                meta = [meta[i] for i in np.where(mask)[0].tolist()]
            Xtr.append(X); ytr.append(y); mtr.extend(meta)

        if out_va is not None:
            X, y, meta = out_va
            Xva.append(X); yva.append(y); mva.extend(meta)

        if out_te is not None:
            X, y, meta = out_te
            Xte.append(X); yte.append(y); mte.extend(meta)

    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        raise RuntimeError("Not enough data after windowing. Try smaller --window.")

    X_train = np.concatenate(Xtr, axis=0)
    y_train = np.concatenate(ytr, axis=0)
    X_val = np.concatenate(Xva, axis=0)
    y_val = np.concatenate(yva, axis=0)
    X_test = np.concatenate(Xte, axis=0)
    y_test = np.concatenate(yte, axis=0)

    if max_train_windows > 0 and len(X_train) > max_train_windows:
        idx = np.random.choice(len(X_train), size=max_train_windows, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        mtr = [mtr[i] for i in idx.tolist()]

    return SplitPack(
        X_train=X_train, y_train=y_train, meta_train=mtr,
        X_val=X_val, y_val=y_val, meta_val=mva,
        X_test=X_test, y_test=y_test, meta_test=mte
    )


# -----------------------------
# F1 BOOST TOOLS (THE IMPORTANT PART)
# -----------------------------
def apply_persistence(pred: np.ndarray, k: int) -> np.ndarray:
    """Keep anomaly only if we have k consecutive anomaly points."""
    if k <= 1:
        return pred.astype(int)
    out = np.zeros_like(pred, dtype=int)
    run = 0
    for i in range(len(pred)):
        if pred[i]:
            run += 1
        else:
            run = 0
        out[i] = 1 if run >= k else 0
    return out


def point_scores_max_overlap(n_points: int, window_scores: np.ndarray, window_ends: np.ndarray, window_len: int) -> np.ndarray:
    """Assign each window score to all points in the window, and take max at each point."""
    scores = np.full(n_points, -np.inf, dtype=float)
    for sc, end in zip(window_scores, window_ends):
        end = int(end)
        start = max(0, end - window_len + 1)
        scores[start:end + 1] = np.maximum(scores[start:end + 1], float(sc))
    if np.isneginf(scores).all():
        return np.zeros(n_points, dtype=float)
    scores[np.isneginf(scores)] = np.nanmin(scores[~np.isneginf(scores)])
    return scores


def point_scores_end_only(n_points: int, window_scores: np.ndarray, window_ends: np.ndarray) -> np.ndarray:
    s = np.full(n_points, np.nan, dtype=float)
    for sc, end in zip(window_scores, window_ends):
        s[int(end)] = float(sc)
    if np.isnan(s).all():
        return np.zeros_like(s)
    first = np.nanmin(np.where(~np.isnan(s))[0])
    s[:first] = s[first]
    mask = np.isnan(s)
    if mask.any():
        s[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), s[~mask])
    return s


def split_point_arrays(sd: SeriesData, split: str):
    if split == "train":
        df = sd.df.iloc[:sd.train_end].reset_index(drop=True)
    elif split == "val":
        df = sd.df.iloc[sd.train_end:sd.val_end].reset_index(drop=True)
    elif split == "test":
        df = sd.df.iloc[sd.val_end:].reset_index(drop=True)
    else:
        raise ValueError(split)
    ts = pd.DatetimeIndex(df["timestamp"]).values
    vals = df["value"].values.astype(float)
    labs = df["label"].values.astype(int)
    return ts, vals, labs


def build_point_scores_from_meta(series_list: List[SeriesData], meta: List[Dict[str, Any]],
                                window_scores: np.ndarray, split: str, window_len: int,
                                point_agg: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict:
      series_key -> { "scores": np.ndarray, "labels": np.ndarray, "ts":..., "vals":... }
    """
    per_series_end: Dict[str, List[int]] = {}
    per_series_sc: Dict[str, List[float]] = {}
    for m, sc in zip(meta, window_scores):
        k = m["series_key"]
        per_series_end.setdefault(k, []).append(int(m["end_pos_in_split"]))
        per_series_sc.setdefault(k, []).append(float(sc))

    out: Dict[str, Dict[str, Any]] = {}
    for sd in series_list:
        ts, vals, labs = split_point_arrays(sd, split)
        n = len(vals)
        ends = np.asarray(per_series_end.get(sd.key, []), dtype=int)
        scs = np.asarray(per_series_sc.get(sd.key, []), dtype=float)

        if len(ends) == 0:
            scores = np.zeros(n, dtype=float)
        else:
            if point_agg == "max_overlap":
                scores = point_scores_max_overlap(n, scs, ends, window_len)
            elif point_agg == "end":
                scores = point_scores_end_only(n, scs, ends)
            else:
                raise ValueError("point_agg must be 'max_overlap' or 'end'")

        out[sd.key] = {"scores": scores, "labels": labs, "ts": ts, "vals": vals}
    return out


def compute_train_norm_params(point_dict_train: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    params = {}
    for k, d in point_dict_train.items():
        s = np.asarray(d["scores"], dtype=float)
        mu = float(np.mean(s))
        sd = float(np.std(s) + 1e-9)
        params[k] = (mu, sd)
    return params


def apply_train_z_norm(point_dict: Dict[str, Dict[str, Any]], params: Dict[str, Tuple[float, float]]):
    for k, d in point_dict.items():
        s = np.asarray(d["scores"], dtype=float)
        mu, sd = params.get(k, (float(np.mean(s)), float(np.std(s) + 1e-9)))
        d["scores"] = (s - mu) / sd


def pick_threshold_max_f1_pointwise(point_val: Dict[str, Dict[str, Any]],
                                    persistence_k: int,
                                    n_grid: int = 401,
                                    plot_path: Optional[str] = None) -> Tuple[float, Dict[str, float]]:
    # concat scores for threshold grid only (preds computed per-series)
    all_scores = np.concatenate([d["scores"] for d in point_val.values()], axis=0)
    qs = np.linspace(0.0, 1.0, n_grid)
    thr_grid = np.unique(np.quantile(all_scores, qs))

    best = {"f1": -1.0, "precision": -1.0, "recall": -1.0, "threshold": float(thr_grid[0])}
    f1_curve = []

    for t in thr_grid:
        preds_all = []
        labs_all = []
        for d in point_val.values():
            pred = (d["scores"] >= t).astype(int)
            pred = apply_persistence(pred, persistence_k)
            preds_all.append(pred)
            labs_all.append(d["labels"])
        preds_all = np.concatenate(preds_all, axis=0)
        labs_all = np.concatenate(labs_all, axis=0)

        f1 = f1_score(labs_all, preds_all, zero_division=0)
        prec = precision_score(labs_all, preds_all, zero_division=0)
        rec = recall_score(labs_all, preds_all, zero_division=0)
        f1_curve.append(f1)

        if (f1 > best["f1"] or
            (math.isclose(f1, best["f1"]) and prec > best["precision"]) or
            (math.isclose(f1, best["f1"]) and math.isclose(prec, best["precision"]) and rec > best["recall"]) or
            (math.isclose(f1, best["f1"]) and math.isclose(prec, best["precision"]) and math.isclose(rec, best["recall"]) and t > best["threshold"])):
            best = {"f1": float(f1), "precision": float(prec), "recall": float(rec), "threshold": float(t)}

    if plot_path is not None:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(thr_grid, f1_curve)
        plt.axvline(best["threshold"], linestyle="--")
        plt.title("Validation F1 vs Threshold (point-level, with persistence)")
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    return best["threshold"], best


def eval_pointwise(point_test: Dict[str, Dict[str, Any]], threshold: float, persistence_k: int) -> Dict[str, float]:
    all_scores = np.concatenate([d["scores"] for d in point_test.values()], axis=0)
    all_labels = np.concatenate([d["labels"] for d in point_test.values()], axis=0)

    auroc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else float("nan")

    preds_all = []
    labs_all = []
    for d in point_test.values():
        pred = (d["scores"] >= threshold).astype(int)
        pred = apply_persistence(pred, persistence_k)
        preds_all.append(pred)
        labs_all.append(d["labels"])
    preds_all = np.concatenate(preds_all, axis=0)
    labs_all = np.concatenate(labs_all, axis=0)

    f1 = f1_score(labs_all, preds_all, zero_division=0)
    prec = precision_score(labs_all, preds_all, zero_division=0)
    rec = recall_score(labs_all, preds_all, zero_division=0)
    return {"AUROC": float(auroc), "F1": float(f1), "Precision": float(prec), "Recall": float(rec)}


# -----------------------------
# Models
# -----------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int = 1, hidden: int = 64, latent: int = 16):
        super().__init__()
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.enc_to_latent = nn.Linear(hidden, latent)
        self.latent_to_dec = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(input_size=hidden, hidden_size=n_features, batch_first=True)

    def forward(self, x):
        out, _ = self.encoder(x)
        h_last = out[:, -1, :]
        z = self.enc_to_latent(h_last)

        h0 = self.latent_to_dec(z).unsqueeze(1)
        h0 = h0.repeat(1, x.size(1), 1)
        recon, _ = self.decoder(h0)
        return recon


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float, no_attention: bool):
        super().__init__()
        self.no_attention = no_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = None if no_attention else nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        if not self.no_attention:
            y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
            x = x + y
        z = self.ff(self.norm2(x))
        x = x + z
        return x


class TransformerAutoencoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int = 1, d_model: int = 64,
                 nhead: int = 4, depth: int = 3, d_ff: int = 128,
                 dropout: float = 0.1, use_pos_enc: bool = True, no_attention: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.use_pos_enc = use_pos_enc
        self.in_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout, no_attention=no_attention)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_features)
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x):
        h = self.in_proj(x)
        if self.use_pos_enc:
            h = h + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return self.out_proj(h)


# -----------------------------
# Train & score AE models (with LOSS CURVE HISTORY)
# -----------------------------
def train_autoencoder(model: nn.Module, X_train: np.ndarray, X_val: np.ndarray,
                      device: str, epochs: int, batch_size: int, lr: float) -> Dict[str, List[float]]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val)), batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None

    hist = {"train_mse": [], "val_mse": []}

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon = model(xb)
                loss = loss_fn(recon, xb)
                va_losses.append(loss.item())

        tr = float(np.mean(tr_losses))
        va = float(np.mean(va_losses))
        hist["train_mse"].append(tr)
        hist["val_mse"].append(va)

        print(f"[train] epoch {ep:02d}/{epochs}  train_mse={tr:.6f}  val_mse={va:.6f}")

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


def plot_loss_curve(hist: Dict[str, List[float]], out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(hist["train_mse"]) + 1), hist["train_mse"], label="train_mse")
    plt.plot(range(1, len(hist["val_mse"]) + 1), hist["val_mse"], label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def score_autoencoder(model: nn.Module, X: np.ndarray, device: str, batch_size: int, score_mode: str) -> np.ndarray:
    """
    score_mode:
      - 'mean' : mean MSE over (T,F)  (old)
      - 'max_t': max over time of per-timestep MSE (BEST for F1 usually)
      - 'topk' : mean of top-10% timestep errors
    """
    model.eval()
    model.to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    scores = []
    for (xb,) in loader:
        xb = xb.to(device)
        recon = model(xb)
        per_t = (recon - xb).pow(2).mean(dim=2)  # (B,T)

        if score_mode == "mean":
            err = per_t.mean(dim=1)
        elif score_mode == "max_t":
            err = per_t.max(dim=1).values
        elif score_mode == "topk":
            k = max(1, int(0.10 * per_t.size(1)))
            err = torch.topk(per_t, k=k, dim=1).values.mean(dim=1)
        else:
            raise ValueError("score_mode must be one of: mean, max_t, topk")

        scores.append(err.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


# -----------------------------
# Baselines
# -----------------------------
def run_model_iforest(Xtr: np.ndarray, Xva: np.ndarray, Xte: np.ndarray, seed: int, contamination: str):
    Xtr_f = Xtr.reshape(len(Xtr), -1)
    Xva_f = Xva.reshape(len(Xva), -1)
    Xte_f = Xte.reshape(len(Xte), -1)
    clf = IsolationForest(
        n_estimators=400,
        random_state=seed,
        n_jobs=-1,
        contamination=contamination
    )
    clf.fit(Xtr_f)
    val_scores = -clf.score_samples(Xva_f)
    test_scores = -clf.score_samples(Xte_f)
    train_scores = -clf.score_samples(Xtr_f)
    return train_scores, val_scores, test_scores


def run_model_ocsvm(Xtr: np.ndarray, Xva: np.ndarray, Xte: np.ndarray, nu: float):
    Xtr_f = Xtr.reshape(len(Xtr), -1)
    Xva_f = Xva.reshape(len(Xva), -1)
    Xte_f = Xte.reshape(len(Xte), -1)
    clf = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    clf.fit(Xtr_f)
    train_scores = -clf.decision_function(Xtr_f)
    val_scores = -clf.decision_function(Xva_f)
    test_scores = -clf.decision_function(Xte_f)
    return train_scores, val_scores, test_scores


# -----------------------------
# Failure case visualization (unchanged style)
# -----------------------------
def plot_failure_case(out_path: str, timestamps, values, labels, scores, threshold, center_pos, window_radius, title):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lo = max(0, center_pos - window_radius)
    hi = min(len(values) - 1, center_pos + window_radius)

    ts_seg = timestamps[lo:hi + 1]
    val_seg = values[lo:hi + 1]
    lab_seg = labels[lo:hi + 1]
    scr_seg = scores[lo:hi + 1]

    plt.figure(figsize=(12, 4))
    plt.plot(ts_seg, val_seg, linewidth=1.5)

    anom_idx = np.where(lab_seg == 1)[0]
    if len(anom_idx) > 0:
        plt.scatter(ts_seg[anom_idx], val_seg[anom_idx], marker="x", s=60, label="True anomaly")

    pred_idx = np.where(scr_seg >= threshold)[0]
    if len(pred_idx) > 0:
        plt.scatter(ts_seg[pred_idx], val_seg[pred_idx], marker="o", s=40, label="Pred anomaly")

    plt.axvline(ts_seg[center_pos - lo], linestyle="--", linewidth=1.0)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_f1_bar(baseline_df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(baseline_df["Model"].tolist(), baseline_df["F1"].tolist())
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Test F1")
    plt.title("Test F1 by Model (with F1-boost post-processing)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default="nab_data")
    ap.add_argument("--download", action="store_true")

    ap.add_argument("--max_files", type=int, default=12)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1)

    ap.add_argument("--label_radius", type=int, default=5)
    ap.add_argument("--train_only_normal", action="store_true")

    ap.add_argument("--max_train_windows", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)

    # F1 BOOST knobs
    ap.add_argument("--score_mode", type=str, default="max_t", choices=["mean", "max_t", "topk"])
    ap.add_argument("--point_agg", type=str, default="max_overlap", choices=["end", "max_overlap"])
    ap.add_argument("--persistence_k", type=int, default=3)

    # Baseline knobs
    ap.add_argument("--svm_nu", type=float, default=0.01)
    ap.add_argument("--if_contamination", type=str, default="auto")  # or e.g. "0.01"

    # plots
    ap.add_argument("--plot_curves", action="store_true")

    args = ap.parse_args()
    set_all_seeds(args.seed)

    if args.download:
        repo_root = download_and_extract_nab(args.root)
    else:
        repo_root = os.path.join(args.root, "NAB-master")
        if not os.path.isdir(repo_root):
            raise RuntimeError("NAB not found. Use --download or place NAB-master under --root.")

    data_dir = os.path.join(repo_root, "data")
    labels_path = os.path.join(repo_root, "labels", "combined_labels.json")
    labels = load_nab_labels(labels_path)

    chosen = choose_files_balanced(labels, data_dir, args.max_files)
    print(f"[data] Using {len(chosen)} files (subset):")
    for k in chosen:
        print("  -", k)

    series_list: List[SeriesData] = []
    skipped = 0
    for k in tqdm(chosen, desc="Loading series"):
        try:
            sd = load_one_series(data_dir, k, labels.get(k, []), label_radius=args.label_radius)
            if len(sd.df) < args.window + 20:
                skipped += 1
                continue
            series_list.append(sd)
        except Exception as e:
            skipped += 1
            print(f"[warn] skip {k}: {e}")

    if len(series_list) < 3:
        raise RuntimeError("Too few valid series loaded.")

    print(f"[data] Loaded {len(series_list)} series (skipped {skipped}).")

    pack = prepare_dataset(
        series_list=series_list,
        window=args.window,
        stride=args.stride,
        train_only_normal=args.train_only_normal,
        max_train_windows=args.max_train_windows,
    )

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    results_rows = []

    # ============ Isolation Forest ============
    print("\n=== Baseline: Isolation Forest ===")
    if_tr, if_va, if_te = run_model_iforest(pack.X_train, pack.X_val, pack.X_test, seed=args.seed,
                                           contamination=args.if_contamination)

    # point-level build + train-z normalization
    p_tr = build_point_scores_from_meta(series_list, pack.meta_train, if_tr, "train", args.window, args.point_agg)
    p_va = build_point_scores_from_meta(series_list, pack.meta_val, if_va, "val", args.window, args.point_agg)
    p_te = build_point_scores_from_meta(series_list, pack.meta_test, if_te, "test", args.window, args.point_agg)
    norm_params = compute_train_norm_params(p_tr)
    apply_train_z_norm(p_va, norm_params)
    apply_train_z_norm(p_te, norm_params)

    thr_if, info_if = pick_threshold_max_f1_pointwise(
        p_va, persistence_k=args.persistence_k,
        plot_path="results/plots/iforest_threshold_sweep.png"
    )
    m_if = eval_pointwise(p_te, thr_if, persistence_k=args.persistence_k)
    results_rows.append({"Model": "IsolationForest", "Threshold": thr_if, **m_if, **{f"Val_{k}": v for k, v in info_if.items()}})
    print("  val best:", info_if)
    print("  test:", m_if)

    # ============ One-Class SVM ============
    print("\n=== Baseline: One-Class SVM ===")
    svm_tr, svm_va, svm_te = run_model_ocsvm(pack.X_train, pack.X_val, pack.X_test, nu=args.svm_nu)

    p_tr = build_point_scores_from_meta(series_list, pack.meta_train, svm_tr, "train", args.window, args.point_agg)
    p_va = build_point_scores_from_meta(series_list, pack.meta_val, svm_va, "val", args.window, args.point_agg)
    p_te = build_point_scores_from_meta(series_list, pack.meta_test, svm_te, "test", args.window, args.point_agg)
    norm_params = compute_train_norm_params(p_tr)
    apply_train_z_norm(p_va, norm_params)
    apply_train_z_norm(p_te, norm_params)

    thr_svm, info_svm = pick_threshold_max_f1_pointwise(
        p_va, persistence_k=args.persistence_k,
        plot_path="results/plots/ocsvm_threshold_sweep.png"
    )
    m_svm = eval_pointwise(p_te, thr_svm, persistence_k=args.persistence_k)
    results_rows.append({"Model": "OneClassSVM", "Threshold": thr_svm, **m_svm, **{f"Val_{k}": v for k, v in info_svm.items()}})
    print("  val best:", info_svm)
    print("  test:", m_svm)

    # ============ LSTM AE ============
    print("\n=== Baseline: LSTM Autoencoder ===")
    lstm = LSTMAutoencoder(n_features=1, hidden=64, latent=16)
    hist_lstm = train_autoencoder(lstm, pack.X_train, pack.X_val, device=args.device,
                                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    if args.plot_curves:
        plot_loss_curve(hist_lstm, "results/plots/lstm_loss.png", "LSTM Autoencoder Loss")

    lstm_tr = score_autoencoder(lstm, pack.X_train, device=args.device, batch_size=args.batch_size, score_mode=args.score_mode)
    lstm_va = score_autoencoder(lstm, pack.X_val, device=args.device, batch_size=args.batch_size, score_mode=args.score_mode)
    lstm_te = score_autoencoder(lstm, pack.X_test, device=args.device, batch_size=args.batch_size, score_mode=args.score_mode)

    p_tr = build_point_scores_from_meta(series_list, pack.meta_train, lstm_tr, "train", args.window, args.point_agg)
    p_va = build_point_scores_from_meta(series_list, pack.meta_val, lstm_va, "val", args.window, args.point_agg)
    p_te = build_point_scores_from_meta(series_list, pack.meta_test, lstm_te, "test", args.window, args.point_agg)
    norm_params = compute_train_norm_params(p_tr)
    apply_train_z_norm(p_va, norm_params)
    apply_train_z_norm(p_te, norm_params)

    thr_lstm, info_lstm = pick_threshold_max_f1_pointwise(
        p_va, persistence_k=args.persistence_k,
        plot_path="results/plots/lstm_threshold_sweep.png"
    )
    m_lstm = eval_pointwise(p_te, thr_lstm, persistence_k=args.persistence_k)
    results_rows.append({"Model": "LSTM_AE", "Threshold": thr_lstm, **m_lstm, **{f"Val_{k}": v for k, v in info_lstm.items()}})
    print("  val best:", info_lstm)
    print("  test:", m_lstm)

    # ============ Transformer AE (Main) ============
    print("\n=== Main: Transformer Autoencoder (base) ===")
    base_tr = TransformerAutoencoder(
        seq_len=args.window,
        n_features=1,
        d_model=64,
        nhead=4,
        depth=3,
        d_ff=128,
        dropout=0.1,
        use_pos_enc=True,
        no_attention=False,
    )
    hist_tr = train_autoencoder(base_tr, pack.X_train, pack.X_val, device=args.device,
                                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    if args.plot_curves:
        plot_loss_curve(hist_tr, "results/plots/transformer_loss.png", "Transformer Autoencoder Loss")

    tr_tr = score_autoencoder(base_tr, pack.X_train, device=args.device, batch_size=args.batch_size, score_mode=args.score_mode)
    tr_va = score_autoencoder(base_tr, pack.X_val, device=args.device, batch_size=args.batch_size, score_mode=args.score_mode)
    tr_te = score_autoencoder(base_tr, pack.X_test, device=args.device, batch_size=args.batch_size, score_mode=args.score_mode)

    p_tr = build_point_scores_from_meta(series_list, pack.meta_train, tr_tr, "train", args.window, args.point_agg)
    p_va = build_point_scores_from_meta(series_list, pack.meta_val, tr_va, "val", args.window, args.point_agg)
    p_te = build_point_scores_from_meta(series_list, pack.meta_test, tr_te, "test", args.window, args.point_agg)
    norm_params = compute_train_norm_params(p_tr)
    apply_train_z_norm(p_va, norm_params)
    apply_train_z_norm(p_te, norm_params)

    thr_tr, info_tr = pick_threshold_max_f1_pointwise(
        p_va, persistence_k=args.persistence_k,
        plot_path="results/plots/transformer_threshold_sweep.png"
    )
    m_tr = eval_pointwise(p_te, thr_tr, persistence_k=args.persistence_k)
    results_rows.append({"Model": "Transformer_AE", "Threshold": thr_tr, **m_tr, **{f"Val_{k}": v for k, v in info_tr.items()}})
    print("  val best:", info_tr)
    print("  test:", m_tr)

    # ============ Ablations ============
    ablation_rows = []

    print("\n=== Ablation 1: Remove positional encoding ===")
    no_pos = TransformerAutoencoder(
        seq_len=args.window, n_features=1, d_model=64, nhead=4, depth=3, d_ff=128,
        dropout=0.1, use_pos_enc=False, no_attention=False
    )
    _ = train_autoencoder(no_pos, pack.X_train, pack.X_val, device=args.device,
                          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    np_tr = score_autoencoder(no_pos, pack.X_train, device=args.device, batch_size=args.batch_size, score_mode=args.score_mode)
    np_va = score_autoencoder(no_pos, pack.X_val, device=args.device, batch_size=args.batch_size, score_mode=args.score_mode)
    np_te = score_autoencoder(no_pos, pack.X_test, device=args.device, batch_size=args.batch_size, score_mode=args.score_mode)

    p_tr = build_point_scores_from_meta(series_list, pack.meta_train, np_tr, "train", args.window, args.point_agg)
    p_va = build_point_scores_from_meta(series_list, pack.meta_val, np_va, "val", args.window, args.point_agg)
    p_te = build_point_scores_from_meta(series_list, pack.meta_test, np_te, "test", args.window, args.point_agg)
    norm_params = compute_train_norm_params(p_tr)
    apply_train_z_norm(p_va, norm_params)
    apply_train_z_norm(p_te, norm_params)

    thr_np, _info = pick_threshold_max_f1_pointwise(p_va, persistence_k=args.persistence_k)
    m_np = eval_pointwise(p_te, thr_np, persistence_k=args.persistence_k)
    ablation_rows.append({"Ablation": "NoPositionalEncoding", "Threshold": thr_np, **m_np})

    print("\n=== Ablation 2: Remove attention ===")
    no_attn = TransformerAutoencoder(
        seq_len=args.window, n_features=1, d_model=64, nhead=4, depth=3, d_ff=128,
        dropout=0.1, use_pos_enc=True, no_attention=True
    )
    _ = train_autoencoder(no_attn, pack.X_train, pack.X_val, device=args.device,
                          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    na_tr = score_autoencoder(no_attn, pack.X_train, device=args.device, batch_size=args.batch_size, score_mode=args.score_mode)
    na_va = score_autoencoder(no_attn, pack.X_val, device=args.device, batch_size=args.batch_size, score_mode=args.score_mode)
    na_te = score_autoencoder(no_attn, pack.X_test, device=args.device, batch_size=args.batch_size, score_mode=args.score_mode)

    p_tr = build_point_scores_from_meta(series_list, pack.meta_train, na_tr, "train", args.window, args.point_agg)
    p_va = build_point_scores_from_meta(series_list, pack.meta_val, na_va, "val", args.window, args.point_agg)
    p_te = build_point_scores_from_meta(series_list, pack.meta_test, na_te, "test", args.window, args.point_agg)
    norm_params = compute_train_norm_params(p_tr)
    apply_train_z_norm(p_va, norm_params)
    apply_train_z_norm(p_te, norm_params)

    thr_na, _info = pick_threshold_max_f1_pointwise(p_va, persistence_k=args.persistence_k)
    m_na = eval_pointwise(p_te, thr_na, persistence_k=args.persistence_k)
    ablation_rows.append({"Ablation": "NoAttention", "Threshold": thr_na, **m_na})

    baseline_df = pd.DataFrame(results_rows).sort_values(["F1", "AUROC"], ascending=False)
    ablation_df = pd.DataFrame(ablation_rows).sort_values(["F1", "AUROC"], ascending=False)

    baseline_df.to_csv("results/baseline_results.csv", index=False)
    ablation_df.to_csv("results/ablation_results.csv", index=False)

    print("\n=== Baseline comparison (POINT-LEVEL + PERSISTENCE + OVERLAP MAX) ===")
    print(baseline_df[["Model", "AUROC", "F1", "Precision", "Recall", "Threshold"]].to_string(index=False))

    print("\n=== Ablations ===")
    print(ablation_df[["Ablation", "AUROC", "F1", "Precision", "Recall", "Threshold"]].to_string(index=False))

    # plot model F1 bar
    if args.plot_curves:
        plot_f1_bar(baseline_df, "results/plots/model_f1_bar.png")

    # Failure cases for Transformer (use point scores, not window end-only)
    print("\n=== Failure cases (Transformer) ===")
    # pick 1 series with most points for plotting
    # build per-series point arrays already in p_te
    # find FP/FN indices globally per-series
    os.makedirs("results/failure_cases", exist_ok=True)

    fp_saved = 0
    fn_saved = 0
    for series_key, d in p_te.items():
        scores = d["scores"]
        labels_ = d["labels"]
        ts = d["ts"]
        vals = d["vals"]

        pred = apply_persistence((scores >= thr_tr).astype(int), args.persistence_k)
        fp_idx = np.where((labels_ == 0) & (pred == 1))[0]
        fn_idx = np.where((labels_ == 1) & (pred == 0))[0]

        # take most confident
        if fp_saved < 3 and len(fp_idx) > 0:
            pick = fp_idx[np.argsort(-scores[fp_idx])][: (3 - fp_saved)]
            for p in pick:
                fp_saved += 1
                out_png = f"results/failure_cases/FalsePositive_{fp_saved}_{series_key.replace('/','_')}.png"
                title = f"FP#{fp_saved} {series_key} score={scores[p]:.4f} thr={thr_tr:.4f}"
                plot_failure_case(out_png, ts, vals, labels_, scores, thr_tr, int(p), args.window * 2, title)

        if fn_saved < 3 and len(fn_idx) > 0:
            pick = fn_idx[np.argsort(scores[fn_idx])][: (3 - fn_saved)]
            for p in pick:
                fn_saved += 1
                out_png = f"results/failure_cases/FalseNegative_{fn_saved}_{series_key.replace('/','_')}.png"
                title = f"FN#{fn_saved} {series_key} score={scores[p]:.4f} thr={thr_tr:.4f}"
                plot_failure_case(out_png, ts, vals, labels_, scores, thr_tr, int(p), args.window * 2, title)

        if fp_saved >= 3 and fn_saved >= 3:
            break

    summary = {
        "dataset": "NAB (subset)",
        "files_used": chosen,
        "files_loaded": [sd.key for sd in series_list],
        "window": args.window,
        "stride": args.stride,
        "label_radius": args.label_radius,
        "train_only_normal": args.train_only_normal,
        "score_mode": args.score_mode,
        "point_agg": args.point_agg,
        "persistence_k": args.persistence_k,
        "threshold_strategy": "Maximize validation F1 on point-level scores (with persistence).",
        "baseline_results": results_rows,
        "ablation_results": ablation_rows,
    }
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[done] Wrote results to ./results/")
    print(" - results/baseline_results.csv")
    print(" - results/ablation_results.csv")
    print(" - results/metrics.json")
    print(" - results/plots/ (loss curves + threshold sweeps + F1 bar)")
    print(" - results/failure_cases/ (FP/FN plots)")


if __name__ == "__main__":
    main()
