"""Build 60-second UNSW-NB15 interaction windows with the same features as cic_windows.py.

Source: rdpahalavan/UNSW-NB15 Network-Flows/UNSW_Flow.parquet (raw UNSW-NB15 flows
with IP addresses and Unix start times). A window is positive when at least half
of its flows are not 'normal'.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "Network-Flows" / "UNSW_Flow.parquet"
CACHE = ROOT / "results" / "unsw_windows.npz"


def build_windows() -> dict:
    cols = ["source_ip", "destination_ip", "destination_port", "stime", "dur", "spkts", "attack_label"]
    df = pd.read_parquet(PARQUET, columns=cols).dropna(subset=["stime", "source_ip", "destination_ip"])
    df["destination_port"] = pd.to_numeric(df["destination_port"], errors="coerce").fillna(0)
    df["win"] = ((df["stime"] - df["stime"].min()) // 60).astype(np.int64)
    df["atk"] = (df["attack_label"].astype(str).str.strip().str.lower() != "normal").astype(np.int8)
    df["cross"] = (df["source_ip"] != df["destination_ip"]).astype(np.int8)
    df["hp"] = (df["destination_port"] >= 10000).astype(np.int8)
    g = df.groupby("win")
    max_out = df.groupby(["win", "source_ip"]).size().groupby("win").max()
    atk_frac = g["atk"].mean()
    wins = np.array(sorted(atk_frac.index), dtype=np.int64)
    return {
        "win": wins,
        "y": (atk_frac >= 0.5).astype(np.int8).loc[wins].to_numpy(),
        "y_any": g["atk"].max().astype(np.int8).loc[wins].to_numpy(),
        "atk_frac": atk_frac.loc[wins].to_numpy(dtype=np.float64),
        "n_edge": g.size().loc[wins].to_numpy(dtype=np.float64),
        "n_src": g["source_ip"].nunique().loc[wins].to_numpy(dtype=np.float64),
        "n_dst": g["destination_ip"].nunique().loc[wins].to_numpy(dtype=np.float64),
        "max_out": max_out.reindex(wins).fillna(0).to_numpy(dtype=np.float64),
        "n_port": g["destination_port"].nunique().loc[wins].to_numpy(dtype=np.float64),
        "cross": g["cross"].mean().loc[wins].to_numpy(dtype=np.float64),
        "dur": g["dur"].median().loc[wins].to_numpy(dtype=np.float64),
        "fwd": g["spkts"].mean().loc[wins].to_numpy(dtype=np.float64),
        "high_port": g["hp"].mean().loc[wins].to_numpy(dtype=np.float64),
        "start_unix": float(df["stime"].min()),
    }


if __name__ == "__main__":
    feat = build_windows()
    np.savez_compressed(CACHE, **feat)
    y, w = feat["y"], feat["win"]
    gaps = np.where(np.diff(w) > 60)[0]
    print("windows", len(y), "pos", int(y.sum()), "pos_rate", round(float(y.mean()), 4),
          "any_rate", round(float(feat["y_any"].mean()), 4), "blocks", len(gaps) + 1,
          "gap_minutes", np.diff(w)[gaps].tolist(), "edges_mean", round(float(feat["n_edge"].mean()), 1))
