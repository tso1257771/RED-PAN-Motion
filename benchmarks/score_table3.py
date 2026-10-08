#!/usr/bin/env python
"""Table III: joint F1 on the six datasets shared with PhaseNet and EQTransformer, threshold 0.3.

Per record the trigger with the highest mean mask (the native baselines have one row per record);
TP = mask_mean >= 0.3, P >= 0.3, S >= 0.3 with |dP| <= 0.5 s and |dS| <= 1.0 s; FP = the same rule
on the SAME dataset's noise records. STEAD: the 16,301 test.npy earthquakes not in the 90 s
training/validation split, and the 23,526 official test noise records (for the RED-PAN models from
the ``stead_noise_test`` run; the baselines' ``stead`` noise rows are the same 60 s traces).
GeoNet: the 70,583 earthquakes of the 2013-2014 holdout and, by default, the 16,384 noise records of
the GeoNet test split (2024; ``--geonet-noise holdout`` uses the holdout's 22,479 noise records, as
the first submission did). CREW and ROMPLUS have no noise, so their F1 reflects recall; CREW recall =
F1 / (2 - F1).

``--pick-f1`` adds P and S pick F1 at the same threshold with no detection gate, by the rule of
``rpm_bench/picks.py`` for every model (the scored pick is the highest peak near the label on an
earthquake and the highest peak of the record on noise), on the same records. The baselines' rows
already hold that pick; the RED-PAN models need ``run_static.py --picks`` (and
``run_static_noise.py --picks`` for the GeoNet test noise), which write ``<dataset>_picks.csv``.

Inputs: static/<model>/{stead,stead_noise_test,geonet,geonet_test_noise,crew,tw,instance,romplus}.csv
(RED-PAN models; ``_picks.csv`` for --pick-f1) and
native/<model>/{stead,geonet,geonet_noise_test,crew,tw,instance,romplus}.csv (baselines).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rpm_bench import config
from rpm_bench.constants import COMMON_THR, P_TOL_SEC, S_TOL_SEC
from rpm_bench.picks import pick_f1
from rpm_bench.results import (
    SHARED_SETS,
    native_csv,
    picks_csv,
    static_csv,
    stead_heldout,
    write_json,
)

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
PICK_COLS = [c for c in COLS if c != "mask_mean"]


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


def records(cfg, m, clean, noise, geonet_noise, picks=False) -> dict:
    """{dataset: (earthquake rows, noise rows)}, one row per record, for one model. ``picks``: the
    rows that carry the scored picks of rpm_bench/picks.py (RED-PAN ``_picks.csv``; the native rows
    of the baselines) instead of the highest-mean-trigger rows."""
    redpan = m in REDPAN_FAMILY
    if redpan:
        path = (lambda ds: picks_csv(cfg, m, ds)) if picks else (lambda ds: static_csv(cfg, m, ds))
    else:
        path = lambda ds: native_csv(cfg, m, ds)  # noqa: E731
    cols = PICK_COLS if (picks and redpan) else COLS

    def read(ds):
        d = pd.read_csv(path(ds), usecols=cols, low_memory=False)
        d = d.drop_duplicates("evid") if (picks and redpan) else pt(d)
        d["evid"] = d.evid.astype(str)
        return d

    out = {}
    s = read("stead")
    eq = s[(s.label_type == "earthquake") & s.evid.isin(clean)]
    if redpan:
        n2 = read("stead_noise_test")
        nz = n2[n2.evid.isin(noise)]
    else:
        nz = s[(s.label_type == "noise") & s.evid.isin(noise)]
    out["stead"] = (eq, nz)
    g = read("geonet")
    if geonet_noise == "test":
        gn = read("geonet_test_noise" if redpan else "geonet_noise_test")
    else:
        gn = g
    out["geonet"] = (g[g.label_type == "earthquake"], gn[gn.label_type == "noise"])
    for ds in SHARED_SETS[2:]:
        d = read(ds)
        out[ds] = (d[d.label_type == "earthquake"], d[d.label_type == "noise"])
    return out


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
    p.add_argument(
        "--geonet-noise",
        choices=["test", "holdout"],
        default="test",
        help="GeoNet false positives on the GeoNet test-split noise (16,384, default) or on the "
        "holdout's 22,479 noise records",
    )
    p.add_argument(
        "--pick-f1",
        action="store_true",
        help="also score P and S pick F1 (rpm_bench/picks.py) on the same records",
    )
    p.add_argument("--out-json", type=Path, default=None, help="also write the scores as JSON")
    args = p.parse_args()
    cfg = config.load(args.config)
    clean, noise = stead_heldout(cfg)
    res = {}
    for m in args.models:
        recs = records(cfg, m, clean, noise, args.geonet_noise)
        row = {ds: joint(*recs[ds]) for ds in SHARED_SETS}
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
    if args.pick_f1:
        pk = {}
        for m in args.models:
            recs = records(cfg, m, clean, noise, args.geonet_noise, picks=True)
            pk[m] = {
                ds: {ph: pick_f1(*recs[ds], ph, COMMON_THR) for ph in "PS"} for ds in SHARED_SETS
            }
            for ph in "PS":
                pk[m][f"macro_{ph}"] = float(np.mean([pk[m][ds][ph]["f1"] for ds in SHARED_SETS]))
        for ph in "PS":
            print(f"\n{ph} pick F1 (threshold {COMMON_THR}, no detection gate)")
            print(f"{'model':18s} " + " ".join(f"{d:>9s}" for d in SHARED_SETS) + "    macro")
            for m, r in pk.items():
                print(
                    f"{m:18s} "
                    + " ".join(f"{r[d][ph]['f1']:9.4f}" for d in SHARED_SETS)
                    + f"  {r[f'macro_{ph}']:7.4f}"
                )
        for m in args.models:
            res[m]["pick_f1"] = pk[m]
    if args.out_json:
        write_json(args.out_json, res)


if __name__ == "__main__":
    main()
