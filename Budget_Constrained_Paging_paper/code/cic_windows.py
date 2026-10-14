"""Build 60-second CIC-IDS2017 interaction windows and cache them.

Nodes are source/destination IPs. A window is labelled attack if at least
half of its flows are not BENIGN (attack-dominant minute). The lexical verifier uses destination-port diversity, which
is withheld from the neural risk head.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "Network-Flows" / "CICIDS_Flow.parquet"
CACHE = ROOT / "results" / "cic_windows.npz"


def _parse_ts(s: pd.Series) -> pd.Series:
    raw = s.astype(str)
    t = pd.to_datetime(raw, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    m = t.isna()
    if m.any():
        t.loc[m] = pd.to_datetime(raw[m], format="%d/%m/%Y %H:%M", errors="coerce")
    m = t.isna()
    if m.any():
        t.loc[m] = pd.to_datetime(raw[m], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    m = t.isna()
    if m.any():
        t.loc[m] = pd.to_datetime(raw[m], format="%m/%d/%Y %H:%M", errors="coerce")
    m = t.isna()
    if m.any():
        t.loc[m] = pd.to_datetime(raw[m], dayfirst=True, errors="coerce")
    # CIC-IDS2017 stores afternoon times on a 12-hour clock without AM/PM.
    # Capture ran 08:00-17:00, so hours 1-7 are afternoon hours 13-19.
    pm = t.dt.hour.between(1, 7)
    t.loc[pm] = t.loc[pm] + pd.Timedelta(hours=12)
    return t


def build_windows(max_rows: int | None = None) -> dict:
    cols = [
        "source_ip",
        "destination_ip",
        "destination_port",
        "Timestamp",
        "attack_label",
        "Flow Duration",
        "Total Fwd Packets",
    ]
    df = pd.read_parquet(PARQUET, columns=cols)
    if max_rows is not None:
        df = df.iloc[:max_rows].copy()
    df["ts"] = _parse_ts(df["Timestamp"])
    df = df.dropna(subset=["ts", "source_ip", "destination_ip"])
    t0 = df["ts"].min()
    df["win"] = ((df["ts"] - t0).dt.total_seconds() // 60).astype(np.int32)
    df["atk"] = (df["attack_label"].astype(str).str.upper() != "BENIGN").astype(np.int8)
    df["cross"] = (df["source_ip"] != df["destination_ip"]).astype(np.int8)

    outdeg = df.groupby(["win", "source_ip"], sort=False).size().reset_index(name="od")
    max_out = outdeg.groupby("win")["od"].max()
    n_src = df.groupby("win")["source_ip"].nunique()
    n_dst = df.groupby("win")["destination_ip"].nunique()
    n_edge = df.groupby("win").size()
    n_port = df.groupby("win")["destination_port"].nunique()
    cross = df.groupby("win")["cross"].mean()
    dur = df.groupby("win")["Flow Duration"].median()
    fwd = df.groupby("win")["Total Fwd Packets"].mean()
    atk_frac = df.groupby("win")["atk"].mean()
    y_any = df.groupby("win")["atk"].max().astype(np.int8)
    y = (atk_frac >= 0.50).astype(np.int8)
    high_ports = (
        df.assign(hp=(df["destination_port"] >= 10000).astype(np.int8))
        .groupby("win")["hp"]
        .mean()
    )
    wins = np.array(sorted(y.index), dtype=np.int32)
    feat = {
        "win": wins,
        "y": y.loc[wins].to_numpy(),
        "y_any": y_any.loc[wins].to_numpy(),
        "atk_frac": atk_frac.loc[wins].to_numpy(dtype=np.float64),
        "n_edge": n_edge.loc[wins].to_numpy(dtype=np.float64),
        "n_src": n_src.loc[wins].to_numpy(dtype=np.float64),
        "n_dst": n_dst.loc[wins].to_numpy(dtype=np.float64),
        "max_out": max_out.reindex(wins).fillna(0).to_numpy(dtype=np.float64),
        "n_port": n_port.loc[wins].to_numpy(dtype=np.float64),
        "cross": cross.loc[wins].to_numpy(dtype=np.float64),
        "dur": dur.loc[wins].to_numpy(dtype=np.float64),
        "fwd": fwd.loc[wins].to_numpy(dtype=np.float64),
        "high_port": high_ports.loc[wins].to_numpy(dtype=np.float64),
    }
    return feat


def save_cache(feat: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **feat)


def load_cache() -> dict:
    z = np.load(CACHE)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    feat = build_windows()
    save_cache(feat)
    y = feat["y"]
    print("windows", len(y), "attack_frac", float(y.mean()), "n_edge_mean", float(feat["n_edge"].mean()))
