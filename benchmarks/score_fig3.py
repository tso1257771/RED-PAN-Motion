#!/usr/bin/env python
"""Fig. 3: joint F1 against catalog S-P time, thresholds fit per 5 s bin (``f1_vs_sp_time.csv``).

Earthquakes: the six shared datasets, held-out records only (STEAD: the 16,301 test.npy earthquakes
not in the 90 s training/validation split); per record the trigger with the highest mean mask.
For each bin, sweep mask 0.1-0.7, P 0.1-0.5, S 0.1-0.5 and keep the best F1; recall over the bin's
earthquakes, precision from the pooled-noise false positives (92,213 records; per record the
highest-mean trigger, same joint rule). Baselines (EQTransformer) use the same pooled noise: their
STEAD / INSTANCE / TW noise rows plus runs on the GeoNet and RockNet test noise.

Inputs: static/<model>/{stead,geonet,crew,tw,instance,romplus}.csv + noise/<model>/noisefp_triggers_<POOL>.csv;
native/<eqt>/{stead,geonet,crew,tw,instance,romplus,geonet_noise_test,rocknet_noise_test}.csv.
Writes ``f1_vs_sp_time.csv`` (model, bin, n, mask, P, S, recall, F1), plotted by plot_fig3.py. A bin
without earthquakes gets n = 0 and empty thresholds and scores (with a warning).
"""

from __future__ import annotations

import argparse
import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from rpm_bench import config
from rpm_bench.constants import P_TOL_SEC as TOL_P, S_TOL_SEC as TOL_S
from rpm_bench.results import POOLS, SHARED_SETS, native_csv, noise_csv, static_csv, stead_heldout

EDGES = [0, 5, 10, 15, 20, 25, 30, np.inf]
LABELS = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", ">30"]
MASK_THR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
P_THR = [0.1, 0.2, 0.3, 0.4, 0.5]
S_THR = [0.1, 0.2, 0.3, 0.4, 0.5]
COLS = [
    "evid",
    "label_type",
    "ps_diff_sec",
    "mask_mean",
    "P_pick_prob",
    "S_pick_prob",
    "P_residual_sec",
    "S_residual_sec",
]
LABEL = {
    "edge": "EdgeRP90",
    "rpm": "RED-PAN-Motion",
    "redpan": "RED-PAN 60s",
    "eqt_stead": "EQTransformer (STEAD)",
    "eqt_instance": "EQTransformer (INSTANCE)",
}


def per_trace(df, key="evid"):
    """The highest-mean-mask row of each ``key``."""
    return df.sort_values([key, "mask_mean"], ascending=[True, False]).drop_duplicates(
        key, keep="first"
    )


def load_eq(paths, clean) -> pd.DataFrame:
    """Best-trigger earthquake rows of the shared sets (STEAD: held-out records only)."""
    eqs = []
    for ds, f in paths.items():
        d = per_trace(pd.read_csv(f, usecols=COLS, low_memory=False))
        e = d[d.label_type == "earthquake"]
        eqs.append(e[e.evid.astype(str).isin(clean)] if ds == "stead" else e)
    return pd.concat(eqs, ignore_index=True)


def pooled_noise_redpan(cfg, model) -> pd.DataFrame:
    """Best-trigger rows of the five pooled-noise sets (untriggered records: mask_mean 0)."""
    out = []
    for pool in POOLS:
        d = pd.read_csv(
            noise_csv(cfg, model, pool),
            usecols=["hdf5_index", "trigger_idx", "mask_mean", "inside_max_P", "inside_max_S"],
        )
        b = per_trace(d, "hdf5_index")
        b = b.assign(mask_mean=b.mask_mean.where(b.trigger_idx >= 0, 0.0))
        out.append(b.rename(columns={"inside_max_P": "P_pick_prob", "inside_max_S": "S_pick_prob"}))
    return pd.concat(out, ignore_index=True)


def pooled_noise_native(cfg, model) -> pd.DataFrame:
    """Noise rows of a baseline over the same five pools."""
    fs = [
        native_csv(cfg, model, d)
        for d in ("stead", "instance", "tw", "geonet_noise_test", "rocknet_noise_test")
    ]
    nz = [per_trace(pd.read_csv(f, usecols=COLS)) for f in fs]
    return pd.concat([d[d.label_type == "noise"] for d in nz], ignore_index=True)


def sweep(label, eq, nz) -> list:
    """One row per S-P bin: the (mask, P, S) with the best F1 in that bin, its recall and F1."""
    eq = eq[eq.ps_diff_sec > 0].copy()
    eq["bin"] = pd.cut(eq.ps_diff_sec, bins=EDGES, labels=LABELS, right=False)
    intol = (eq.P_residual_sec.abs() <= TOL_P) & (eq.S_residual_sec.abs() <= TOL_S)
    n_by = eq.groupby("bin", observed=False).size().reindex(LABELS).fillna(0).astype(int)
    best = {b: None for b in LABELS}
    for m, p, s in product(MASK_THR, P_THR, S_THR):
        ok = (eq.mask_mean >= m) & (eq.P_pick_prob >= p) & (eq.S_pick_prob >= s) & intol
        fp = int(((nz.mask_mean >= m) & (nz.P_pick_prob >= p) & (nz.S_pick_prob >= s)).sum())
        tp_by = ok.groupby(eq.bin, observed=False).sum().reindex(LABELS).fillna(0).astype(int)
        for b in LABELS:
            n_b, tp = int(n_by[b]), int(tp_by[b])
            if n_b == 0:
                continue
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / n_b
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            if best[b] is None or f1 > best[b][3]:
                best[b] = (m, p, s, f1, rec)
    empty = [b for b in LABELS if best[b] is None]
    if empty:
        logging.warning(
            "%s: no earthquakes in S-P bin(s) %s; their thresholds and scores are left empty",
            label,
            ", ".join(empty),
        )
    nan5 = (np.nan,) * 5
    return [
        {
            "model": label,
            "bin": b,
            "n": int(n_by[b]),
            "mask": (best[b] or nan5)[0],
            "P": (best[b] or nan5)[1],
            "S": (best[b] or nan5)[2],
            "recall": (best[b] or nan5)[4],
            "F1": (best[b] or nan5)[3],
        }
        for b in LABELS
    ]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument(
        "--models", nargs="+", default=["edge", "rpm", "redpan", "eqt_stead", "eqt_instance"]
    )
    p.add_argument(
        "--out", type=Path, default=None, help="default <results_root>/f1_vs_sp_time.csv"
    )
    args = p.parse_args()
    cfg = config.load(args.config)
    clean, _ = stead_heldout(cfg)
    rows = []
    for m in args.models:
        native = m.startswith(("eqt_", "phasenet_"))
        paths = {ds: (native_csv if native else static_csv)(cfg, m, ds) for ds in SHARED_SETS}
        nz = pooled_noise_native(cfg, m) if native else pooled_noise_redpan(cfg, m)
        rows += sweep(LABEL.get(m, m), load_eq(paths, clean), nz)
    tab = pd.DataFrame(rows)
    out = args.out or cfg["results_root"] / "f1_vs_sp_time.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)
    print(
        tab.pivot(index="model", columns="bin", values="F1")
        .reindex(columns=LABELS)
        .round(4)
        .to_string()
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
