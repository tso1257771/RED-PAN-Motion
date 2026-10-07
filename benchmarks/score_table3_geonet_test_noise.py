#!/usr/bin/env python
"""Table III with the GeoNet false positives counted on the GeoNet TEST noise.

``score_table3.py`` counts GeoNet false positives on the noise of the 2013-2014 holdout archive
(22,479 records), which overlaps the 90 s training and validation noise. This scorer keeps
everything else of Table III (same rule: highest-mean trigger per record, threshold 0.3, |dP| <= 0.5 s,
|dS| <= 1.0 s; same GeoNet holdout earthquakes) and replaces only the GeoNet noise by the noise test
split of the 90 s GeoNet archive (16,384 records, the GeoNet pool of the pooled noise set, disjoint
from the training and validation splits). It prints the old and new GeoNet cells, the macro F1
over the six shared sets and the best score per row (bold in the manuscript).

Inputs, in addition to those of ``score_table3.py``:
  static/<model>/geonet_test_noise.csv   (RED-PAN models, ``run_static_noise.py --pool GeoNet``)
  native/<model>/geonet_noise_test.csv   (baselines, ``run_native.py --dataset geonet_noise_test``)

Example:
    python score_table3_geonet_test_noise.py --out-json results/table3_geonet_test_noise.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from rpm_bench import config
from rpm_bench.results import SHARED_SETS, native_csv, static_csv, stead_heldout, write_json
from score_table3 import COLS, REDPAN_FAMILY, joint, pt

MODELS = [
    "edge",
    "rpm",
    "redpan",
    "phasenet_stead",
    "phasenet_instance",
    "eqt_stead",
    "eqt_instance",
]


def _read(path):
    d = pt(pd.read_csv(path, usecols=COLS, low_memory=False))
    d["evid"] = d.evid.astype(str)
    return d


def score_model(cfg, m, clean, noise):
    """Table III cells for one model; ``geonet`` = holdout noise (old), ``geonet_test_noise`` = new."""
    redpan = m in REDPAN_FAMILY
    path = (lambda ds: static_csv(cfg, m, ds)) if redpan else (lambda ds: native_csv(cfg, m, ds))
    row = {}
    s = _read(path("stead"))
    eq = s[(s.label_type == "earthquake") & s.evid.isin(clean)]
    if redpan:
        n2 = _read(static_csv(cfg, m, "stead_noise_test"))
        nz = n2[n2.evid.isin(noise)]
    else:
        nz = s[(s.label_type == "noise") & s.evid.isin(noise)]
    row["stead"] = joint(eq, nz)
    for ds in SHARED_SETS[1:]:
        d = _read(path(ds))
        row[ds] = joint(d[d.label_type == "earthquake"], d[d.label_type == "noise"])
    g = _read(path("geonet"))
    gnz = _read(
        static_csv(cfg, m, "geonet_test_noise")
        if redpan
        else native_csv(cfg, m, "geonet_noise_test")
    )
    row["geonet_test_noise"] = joint(
        g[g.label_type == "earthquake"], gnz[gnz.label_type == "noise"]
    )
    row["macro_old"] = float(np.mean([row[ds]["f1"] for ds in SHARED_SETS]))
    row["macro_new"] = float(
        np.mean([row["geonet_test_noise" if ds == "geonet" else ds]["f1"] for ds in SHARED_SETS])
    )
    return row


def best(res, key):
    vals = {m: (r[key]["f1"] if isinstance(r[key], dict) else r[key]) for m, r in res.items()}
    top = max(round(v, 3) for v in vals.values())
    return [m for m, v in vals.items() if round(v, 3) == top]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()
    cfg = config.load(args.config)
    clean, noise = stead_heldout(cfg)
    res = {m: score_model(cfg, m, clean, noise) for m in args.models}

    print(
        f"{'model':18s} {'GeoNet old':>10s} {'FP old':>7s} {'GeoNet new':>10s} {'FP new':>7s} "
        f"{'R':>7s} {'macro old':>9s} {'macro new':>9s}"
    )
    for m, r in res.items():
        o, n = r["geonet"], r["geonet_test_noise"]
        print(
            f"{m:18s} {o['f1']:10.4f} {o['fp']:7d} {n['f1']:10.4f} {n['fp']:7d} {n['recall']:7.4f} "
            f"{r['macro_old']:9.4f} {r['macro_new']:9.4f}"
        )
    any_m = next(iter(res.values()))
    print(
        f"GeoNet n: {any_m['geonet']['n_eq']} earthquakes; noise {any_m['geonet']['n_noise']} (holdout) -> "
        f"{any_m['geonet_test_noise']['n_noise']} (test split)"
    )
    marks = {
        "GeoNet old": best(res, "geonet"),
        "GeoNet new": best(res, "geonet_test_noise"),
        "macro old": best(res, "macro_old"),
        "macro new": best(res, "macro_new"),
    }
    print("best per row (3 decimals):", marks)
    if args.out_json:
        write_json(args.out_json, {"scores": res, "best": marks})


if __name__ == "__main__":
    main()
