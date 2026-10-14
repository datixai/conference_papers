"""Verify the 12-hour clock repair and the attack schedule on the raw CIC-IDS2017 flow export."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cic_windows import PARQUET, ROOT, _parse_ts, build_windows, load_cache
from run_revision import repair_clock


def main() -> None:
    df = pd.read_parquet(PARQUET, columns=["Timestamp", "attack_label"])
    raw = df["Timestamp"].astype(str)
    out = {"rows": int(len(df)), "timestamp_examples": raw.sample(5, random_state=0).tolist()}
    hour = pd.to_numeric(raw.str.extract(r"\s(\d{1,2}):")[0], errors="coerce")
    out["raw_hour_counts"] = {int(k): int(v) for k, v in hour.value_counts().sort_index().items()}
    df["ts"] = _parse_ts(df["Timestamp"])
    out["unparsed"] = int(df["ts"].isna().sum())
    df = df.dropna(subset=["ts"])
    df["day"] = df["ts"].dt.strftime("%a %d %b")
    df["lab"] = df["attack_label"].astype(str)
    sched = (
        df[df["lab"].str.upper() != "BENIGN"]
        .groupby(["day", "lab"])["ts"]
        .agg(["min", "max", "count"])
        .reset_index()
        .sort_values("min")
    )
    out["attack_schedule"] = [
        {"day": r.day, "label": r.lab, "start": r["min"].strftime("%H:%M"), "end": r["max"].strftime("%H:%M"), "flows": int(r["count"])}
        for _, r in sched.iterrows()
    ]
    # Windows rebuilt with the repaired parser must equal the repaired cache.
    fresh = build_windows()
    cached = repair_clock(load_cache())
    fk = {int(w): i for i, w in enumerate(fresh["win"] - fresh["win"].min())}
    ck = {int(w): i for i, w in enumerate(cached["win"] - cached["win"].min())}
    common = sorted(set(fk) & set(ck))
    fi, ci = [fk[w] for w in common], [ck[w] for w in common]
    out["rebuilt_vs_repaired_cache"] = {
        "fresh_windows": len(fk), "cache_windows": len(ck), "common": len(common),
        "only_cache_edges": [float(cached["n_edge"][ck[w]]) for w in set(ck) - set(fk)],
        "labels_equal_on_common": bool(np.array_equal(fresh["y"][fi], cached["y"][ci])),
        "edges_equal_on_common": int((fresh["n_edge"][fi] == cached["n_edge"][ci]).sum()),
        "fresh_pos": int(fresh["y"].sum()),
    }
    # Dominant attack label of each attack-dominant window.
    t0 = df["ts"].min()
    df["win"] = ((df["ts"] - t0).dt.total_seconds() // 60).astype(int)
    atk = df[df["lab"].str.upper() != "BENIGN"]
    frac = df.assign(a=(df["lab"].str.upper() != "BENIGN")).groupby("win")["a"].mean()
    pos = frac[frac >= 0.5].index
    dom = atk[atk["win"].isin(pos)].groupby("win")["lab"].agg(lambda s: s.value_counts().index[0])
    out["positive_windows_by_label"] = {k: int(v) for k, v in dom.value_counts().items()}
    (ROOT / "results" / "cic_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
