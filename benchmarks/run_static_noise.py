#!/usr/bin/env python
"""Static-protocol rows (as run_static.py) for the noise-only test split of a 90 s HDF5 archive.

Used for the Table III GeoNet column with the GeoNet TEST noise (16,384 records, the GeoNet pool of
the pooled noise) instead of the holdout-archive noise (22,479 records, which overlap the 90 s training
and validation splits). Records are preprocessed with the model's filter (after demean and a 5%
taper) and centre-cropped to the model window (the 60 s RED-PAN reads samples 1500:7500), exactly as
run_static.py treats the INSTANCE and TW noise.

Writes ``<results_root>/static/<model>/<pool>_test_noise.csv`` (e.g. ``geonet_test_noise.csv``).
The SeisBench baselines use ``run_native.py --dataset geonet_noise_test``.

Example:
    python run_static_noise.py --model edge --pool GeoNet
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from rpm_bench import config, datasets, models
from rpm_bench.constants import DT
from rpm_bench.picks import record_picks
from rpm_bench.static import build_rows, extract_triggers, run_static


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument("--model", required=True, choices=models.MODEL_KEYS)
    p.add_argument(
        "--pool",
        default="GeoNet",
        choices=["GeoNet", "INSTANCE", "RockNet", "TW"],
        help="archive directory under h5_root whose noise test split is used",
    )
    p.add_argument(
        "--filter",
        default=None,
        choices=["bp145", "hp1", "raw"],
        help="default: the model's filter",
    )
    p.add_argument("--max-noise", type=int, default=0, help="cap (0 = all)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--picks",
        action="store_true",
        help="also write the scored P and S picks per record (rpm_bench/picks.py) to "
        "<out stem>_picks.csv, for the Table III pick F1",
    )
    p.add_argument("--device", default=models.default_device())
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config.load(args.config)
    filt = args.filter or models.FILTER[args.model]
    model = models.load_model(args.model, cfg, args.device)
    out = (
        args.out
        or config.results_dir(cfg, "static", args.model) / f"{args.pool.lower()}_test_noise.csv"
    )
    rows, pick_rows, n, t0 = [], [], 0, time.time()
    for wf, info in datasets.iter_h5_dataset(
        cfg["h5_root"] / args.pool, "noise", args.max_noise, filt=filt
    ):
        mask, p_arr, s_arr, pol = run_static(
            model, wf, model.in_samples, args.device, p_abs=None, s_abs=None
        )
        triggers = extract_triggers(
            mask, p_arr, s_arr, pol, dt=DT, thr_on=0.1, thr_off=0.1, smooth_npts=10
        )
        rec = dict(
            evid=info["evid"], label_type="noise", mode="static", model=models.LABEL[args.model]
        )
        rows.extend(build_rows(rec, triggers, info["labelP_sec"], info["labelS_sec"], ""))
        if args.picks:
            pick_rows.append(record_picks({**info, "label_type": "noise"}, p_arr, s_arr))
        n += 1
        if n % 2000 == 0:
            logging.info("  %d records (%.1fs)", n, time.time() - t0)
    if n == 0:
        raise config.no_records_error(cfg["h5_root"] / args.pool, "h5_root")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    if args.picks:
        pick_out = out.with_name(out.stem + "_picks.csv")
        pd.DataFrame(pick_rows).to_csv(pick_out, index=False)
        logging.info("wrote %d pick rows -> %s", len(pick_rows), pick_out)
    logging.info(
        "wrote %d rows for %d noise records -> %s (%.1fs)", len(rows), n, out, time.time() - t0
    )


if __name__ == "__main__":
    main()
