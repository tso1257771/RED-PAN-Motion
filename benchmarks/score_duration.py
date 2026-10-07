#!/usr/bin/env python
"""Section II-E: Edge-RED-PAN-Motion mask-trigger duration against catalog S-P time.

Triggers: every trigger row of every earthquake record (mask smoothed over 10 samples,
trigger_onset 0.1, as written by run_static.py) with catalog S-P > 0, on the held-out sets of
Table IV (static/edge/{ceed_nc,ceed_sc,crew,geonet,instance,romplus,stead_h5,tw}.csv).
Reports the number of triggers, Pearson and Spearman correlation of duration (end - start) with
S-P, and the median S-P per duration bin (edges 6, 12, 18, 24 s, as in redpan_motion.sp_thresholds).
``--best-trigger`` restricts to the highest-mean trigger of each record.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rpm_bench import config
from rpm_bench.results import HELDOUT_SETS, static_csv, write_json
from scipy.stats import pearsonr, spearmanr

EDGES = [0, 6, 12, 18, 24, np.inf]
LAB = ["<6", "6-12", "12-18", "18-24", ">=24"]
COLS = [
    "evid",
    "label_type",
    "trigger_idx",
    "trigger_on_sec",
    "trigger_off_sec",
    "mask_mean",
    "ps_diff_sec",
]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument("--model", default="edge")
    p.add_argument(
        "--best-trigger", action="store_true", help="only the highest-mean trigger per record"
    )
    p.add_argument("--out-json", type=Path, default=None, help="also write the results as JSON")
    args = p.parse_args()
    cfg = config.load(args.config)
    frames = []
    for ds in HELDOUT_SETS:
        d = pd.read_csv(static_csv(cfg, args.model, ds), usecols=COLS, low_memory=False)
        d = d[(d.label_type == "earthquake") & (d.trigger_idx >= 0) & (d.ps_diff_sec > 0)].copy()
        d["dur"] = d.trigger_off_sec - d.trigger_on_sec
        d["ds"] = ds
        frames.append(d)
    x = pd.concat(frames, ignore_index=True)
    if args.best_trigger:
        x = x.sort_values(
            ["ds", "evid", "mask_mean"], ascending=[True, True, False]
        ).drop_duplicates(["ds", "evid"])
    b = pd.cut(x.dur, EDGES, labels=LAB, right=False)
    r = {
        "n_triggers": len(x),
        "n_records": int(x[["ds", "evid"]].drop_duplicates().shape[0]),
        "pearson": float(pearsonr(x.dur, x.ps_diff_sec)[0]),
        "spearman": float(spearmanr(x.dur, x.ps_diff_sec)[0]),
        "median_sp": {
            k: float(v)
            for k, v in x.groupby(b, observed=False).ps_diff_sec.median().reindex(LAB).items()
        },
        "count": {k: int(v) for k, v in x.groupby(b, observed=False).size().reindex(LAB).items()},
        "per_dataset_triggers": {k: int(v) for k, v in x.groupby("ds").size().items()},
    }
    print(
        f"triggers={r['n_triggers']:,} records={r['n_records']:,} pearson={r['pearson']:.4f} spearman={r['spearman']:.4f}"
    )
    print("median S-P per duration bin (s):", {k: round(v, 2) for k, v in r["median_sp"].items()})
    print("triggers per bin:", r["count"])
    if args.out_json:
        write_json(args.out_json, r)


if __name__ == "__main__":
    main()
