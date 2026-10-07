#!/usr/bin/env python
"""Table III: joint F1 on the six datasets shared with PhaseNet and EQTransformer, threshold 0.3.

Per record the trigger with the highest mean mask (the native baselines have one row per record);
TP = mask_mean >= 0.3, P >= 0.3, S >= 0.3 with |dP| <= 0.5 s and |dS| <= 1.0 s; FP = the same rule
on the SAME dataset's noise records. STEAD: the 16,301 test.npy earthquakes not in the 90 s
training/validation split, and the 23,526 official test noise records (for the RED-PAN models from
the ``stead_noise_test`` run; the baselines' ``stead`` noise rows are the same 60 s traces).
CREW and ROMPLUS have no noise, so their F1 reflects recall; CREW recall = F1 / (2 - F1).

Inputs: static/<model>/{stead,stead_noise_test,geonet,crew,tw,instance,romplus}.csv (RED-PAN models)
and native/<model>/{stead,geonet,crew,tw,instance,romplus}.csv (baselines).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rpm_bench import config
from rpm_bench.constants import COMMON_THR, P_TOL_SEC, S_TOL_SEC
from rpm_bench.results import SHARED_SETS, native_csv, static_csv, stead_heldout, write_json

REDPAN_FAMILY = ("edge", "rpm", "redpan", "redpan_240107")
COLS = [
    "evid",
    "label_type",
    "mask_mean",
    "P_pick_prob",
    "S_pick_prob",
    "P_residual_sec",
    "S_residual_sec",
]


def pt(d):
    """The highest-mean-mask row of each record."""
    return d.sort_values(["evid", "mask_mean"], ascending=[True, False]).drop_duplicates(
        "evid", keep="first"
    )


def joint(eq, nz, m=COMMON_THR, pp=COMMON_THR, ss=COMMON_THR) -> dict:
    """Joint F1, recall and counts of one dataset (earthquake and noise rows, one per record)."""
    tp = int(
        (
            (eq.mask_mean >= m)
            & (eq.P_pick_prob >= pp)
            & (eq.S_pick_prob >= ss)
            & (eq.P_residual_sec.abs() <= P_TOL_SEC)
            & (eq.S_residual_sec.abs() <= S_TOL_SEC)
        ).sum()
    )
    fp = (
        int(((nz.mask_mean >= m) & (nz.P_pick_prob >= pp) & (nz.S_pick_prob >= ss)).sum())
        if len(nz)
        else 0
    )
    fn = len(eq) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(
        f1=2 * p * r / (p + r) if p + r else 0.0, recall=r, n_eq=len(eq), n_noise=len(nz), fp=fp
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument(
        "--models",
        nargs="+",
        default=[
            "edge",
            "rpm",
            "redpan",
            "phasenet_stead",
            "phasenet_instance",
            "eqt_stead",
            "eqt_instance",
        ],
    )
    p.add_argument("--out-json", type=Path, default=None, help="also write the scores as JSON")
    args = p.parse_args()
    cfg = config.load(args.config)
    clean, noise = stead_heldout(cfg)
    res = {}
    for m in args.models:
        path = (
            (lambda ds: static_csv(cfg, m, ds))
            if m in REDPAN_FAMILY
            else (lambda ds: native_csv(cfg, m, ds))
        )
        row = {}
        s = pt(pd.read_csv(path("stead"), usecols=COLS, low_memory=False))
        s["evid"] = s.evid.astype(str)
        eq = s[(s.label_type == "earthquake") & s.evid.isin(clean)]
        if m in REDPAN_FAMILY:
            n2 = pt(pd.read_csv(static_csv(cfg, m, "stead_noise_test"), usecols=COLS))
            n2["evid"] = n2.evid.astype(str)
            nz = n2[n2.evid.isin(noise)]
        else:
            nz = s[(s.label_type == "noise") & s.evid.isin(noise)]
        row["stead"] = joint(eq, nz)
        for ds in SHARED_SETS[1:]:
            d = pt(pd.read_csv(path(ds), usecols=COLS, low_memory=False))
            row[ds] = joint(d[d.label_type == "earthquake"], d[d.label_type == "noise"])
        row["macro"] = float(np.mean([row[ds]["f1"] for ds in SHARED_SETS]))
        res[m] = row
    print(f"{'model':18s} " + " ".join(f"{d:>9s}" for d in SHARED_SETS) + "    macro")
    for m, row in res.items():
        print(
            f"{m:18s} "
            + " ".join(f"{row[d]['f1']:9.4f}" for d in SHARED_SETS)
            + f"  {row['macro']:7.4f}"
        )
    print("CREW recall:", {m: round(r["crew"]["recall"], 4) for m, r in res.items()})
    print(
        "STEAD n (eq, noise):",
        {m: (r["stead"]["n_eq"], r["stead"]["n_noise"]) for m, r in res.items()},
    )
    if args.out_json:
        write_json(args.out_json, res)


if __name__ == "__main__":
    main()
