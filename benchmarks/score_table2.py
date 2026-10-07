#!/usr/bin/env python
"""Table II: joint F1 of the RED-PAN models on the eight test sets, and the detector-trigger count.

Rule (Methods): per record only the trigger with the highest MEAN mask is scored (mask smoothed
over 10 samples, trigger_onset 0.1 - as written by run_static.py / run_noise.py).
  true positive   that trigger has mean mask >= 0.50, P >= 0.30 within 0.5 s and S >= 0.20
                  within 1.0 s of the labels
  false positive  a pooled-noise record whose single trigger passes the same joint rule
Precision of every dataset uses the pooled noise (92,213 records, 5 pools). Earthquake records:
those common to the compared models; noise records never enter the earthquake denominator.
Detector-trigger count: pooled-noise records whose single trigger has mean mask >= 0.50
(also reported: the older whole-record definition, any trigger_onset 0.3/0.3 trigger with peak >= 0.5).

Inputs: static/<model>/{ceed_nc,ceed_sc,crew,geonet,instance,romplus,stead_h5,tw}.csv (GeoNet: the
2013-2014 holdout earthquakes; its noise rows are not used) and noise/<model>/noisefp[_triggers]_<POOL>.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rpm_bench import config
from rpm_bench.constants import P_TOL_SEC as P_TOL, S_TOL_SEC as S_TOL
from rpm_bench.results import NAME, POOLS, TABLE2_SETS, noise_csv, static_csv, write_json

M_THR, P_THR, S_THR = 0.50, 0.30, 0.20  # the deployment thresholds (mean mask, P, S)


def best(df, key):
    """The highest-mean-mask row of each ``key`` (ties: the first in ``key`` order)."""
    return df.sort_values([key, "mask_mean"], ascending=[True, False]).drop_duplicates(
        key, keep="first"
    )


def noise_stats(cfg, model) -> dict:
    """Pooled-noise record count, joint false positives and detector-trigger counts of a model."""
    n = fp = det = det_old = 0
    for pool in POOLS:
        t = best(pd.read_csv(noise_csv(cfg, model, pool)), "hdf5_index")
        n += len(t)
        trig = t.trigger_idx >= 0
        fp += int(
            (
                trig
                & (t.mask_mean >= M_THR)
                & (t.inside_max_P >= P_THR)
                & (t.inside_max_S >= S_THR)
            ).sum()
        )
        det += int((trig & (t.mask_mean >= M_THR)).sum())
        w = pd.read_csv(noise_csv(cfg, model, pool, triggers=False), usecols=["det_trigger_max"])
        det_old += int((w.det_trigger_max.fillna(0) >= 0.5).sum())
    if n == 0:
        raise SystemExit(
            f"no pooled-noise records for model {model} (noise/{model}/noisefp_triggers_<POOL>.csv)"
        )
    return dict(n=n, joint_fp=fp, detector_triggers=det, detector_triggers_old_definition=det_old)


def eq_ok(path) -> pd.Series:
    """Per earthquake record (index evid): whether its best trigger is a joint true positive."""
    d = pd.read_csv(
        path,
        low_memory=False,
        usecols=[
            "evid",
            "label_type",
            "n_triggers",
            "mask_mean",
            "P_pick_prob",
            "P_residual_sec",
            "S_pick_prob",
            "S_residual_sec",
        ],
    )
    b = best(d[d.label_type == "earthquake"], "evid")
    ok = (
        (b.n_triggers >= 1)
        & (b.mask_mean >= M_THR)
        & (b.P_pick_prob >= P_THR)
        & (b.P_residual_sec.abs() <= P_TOL)
        & (b.S_pick_prob >= S_THR)
        & (b.S_residual_sec.abs() <= S_TOL)
    )
    return pd.Series(ok.values, index=b.evid.values)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument("--models", nargs="+", default=["edge", "rpm", "redpan"])
    p.add_argument("--out-json", type=Path, default=None, help="also write the scores as JSON")
    args = p.parse_args()
    cfg = config.load(args.config)
    res = {"noise": {m: noise_stats(cfg, m) for m in args.models}, "f1": {}, "recall": {}, "n": {}}
    for ds in TABLE2_SETS:
        oks = {m: eq_ok(static_csv(cfg, m, ds)) for m in args.models}
        common = sorted(set.intersection(*[set(o.index) for o in oks.values()]))
        if not common:
            raise SystemExit(
                f"{ds}: no earthquake record is common to the models {args.models} "
                f"(static/<model>/{ds}.csv)"
            )
        res["n"][ds] = len(common)
        for m, o in oks.items():
            tp = int(o.loc[common].sum())
            fp = res["noise"][m]["joint_fp"]
            r = tp / len(common)
            prec = tp / (tp + fp) if tp + fp else 0.0
            res["f1"].setdefault(m, {})[ds] = 2 * prec * r / (prec + r) if prec + r else 0.0
            res["recall"].setdefault(m, {})[ds] = r
    for m in args.models:
        res["f1"][m]["macro"] = float(np.mean([res["f1"][m][d] for d in TABLE2_SETS]))
    print(f"{'dataset':10s} {'n':>8s} " + " ".join(f"{m:>10s}" for m in args.models))
    for ds in list(TABLE2_SETS) + ["macro"]:
        print(
            f"{NAME.get(ds, 'Macro'):10s} {res['n'].get(ds, ''):>8} "
            + " ".join(f"{res['f1'][m][ds]:10.4f}" for m in args.models)
        )
    for m, v in res["noise"].items():
        print(
            f"noise {m:8s} n={v['n']:,} joint FP={v['joint_fp']} ({100 * v['joint_fp'] / v['n']:.2f}%) "
            f"detector-trigger records={v['detector_triggers']} (old definition {v['detector_triggers_old_definition']})"
        )
    if args.out_json:
        write_json(args.out_json, res)


if __name__ == "__main__":
    main()
