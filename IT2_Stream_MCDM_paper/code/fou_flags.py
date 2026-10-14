"""Emit per-channel FOU widths and a telemetry-degradation flag.

Flag rule (authorised): for engine i at cycle t, if max_j (UMF-LMF) > THETA
then raise a channel-noise flag. This is implemented, not only described.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from it2_stream_mcdm import (
    DATA,
    OUT,
    SENSOR_COLS,
    T_START,
    T_STEP,
    collect_windows,
    healthy_stats,
    load_train,
    per_engine,
    rank_batch,
)

THETA_CHANNEL = 0.50  # per-cell FOU; near the 90th percentile on FD001
THETA_ENGINE = 0.65  # max_j FOU; tail of the engine-wise width
SENSOR_NAME = {
    6: "T24",
    7: "T30",
    8: "T50",
    11: "P30",
    12: "Nf",
    13: "Nc",
    15: "Ps30",
    16: "phi",
    17: "NRf",
    18: "NRc",
    21: "htBleed",
    24: "W31",
    25: "W32",
}


def main():
    arr = load_train(DATA)
    eng = per_engine(arr)
    mu, sd = healthy_stats(eng)
    w = np.ones(len(SENSOR_COLS)) / len(SENSOR_COLS)
    max_t = max(int(a[:, 1].max()) for a in eng.values())
    times = list(range(T_START, max_t + 1, T_STEP))

    ch_fracs, eng_fracs = [], []
    examples = []
    widths_all = []
    for t in times:
        items = collect_windows(eng, t)
        if len(items) < 15:
            continue
        from it2_stream_mcdm import _fuzzify_one

        mapped = [_fuzzify_one((uid, m, s, mu, sd, 0.0, 0)) for uid, m, s, _ in items]
        L = np.vstack([x[1] for x in mapped])
        U = np.vstack([x[2] for x in mapped])
        fou = U - L
        widths_all.append(fou.ravel())
        maxw = fou.max(axis=1)
        ch_fracs.append(float((fou > THETA_CHANNEL).mean()))
        eng_fracs.append(float((maxw > THETA_ENGINE).mean()))
        uids = [x[0] for x in mapped]
        _, it2, _, _ = rank_batch(items, mu, sd, w, 0.0, 0, 1)
        if t == 140:
            order = np.argsort(-it2)
            for k in order[:12]:
                j = int(np.argmax(fou[k]))
                examples.append(
                    {
                        "t": t,
                        "engine": int(uids[k]),
                        "C_i": float(it2[k]),
                        "max_FOU": float(maxw[k]),
                        "engine_flag": bool(maxw[k] > THETA_ENGINE),
                        "widest_sensor": SENSOR_NAME[SENSOR_COLS[j]],
                        "interval": [float(L[k, j]), float(U[k, j])],
                    }
                )

    wa = np.concatenate(widths_all)
    out = {
        "theta_channel": THETA_CHANNEL,
        "theta_engine": THETA_ENGINE,
        "rule_channel": "flag cell (i,j) if UMF-LMF > theta_channel",
        "rule_engine": "flag engine i if max_j (UMF-LMF) > theta_engine",
        "n_batches": len(ch_fracs),
        "channel_flag_rate": [float(np.mean(ch_fracs)), float(np.std(ch_fracs, ddof=1))],
        "engine_flag_rate": [float(np.mean(eng_fracs)), float(np.std(eng_fracs, ddof=1))],
        "fou_width_mean": float(wa.mean()),
        "fou_width_p90": float(np.quantile(wa, 0.90)),
        "fou_width_max": float(wa.max()),
        "examples_t140": examples,
    }
    (OUT / "fou_flags.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
