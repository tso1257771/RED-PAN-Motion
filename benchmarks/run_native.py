#!/usr/bin/env python
"""SeisBench PhaseNet / EQTransformer baselines in their native configuration (Table III, Fig. 3).

Writes ``<results_root>/native/<model>/<dataset>.csv`` in the static-run schema (one row per
record; ``mask_peak`` = ``mask_mean`` = the record's detection peak, picks = the best peak inside
the tolerance window). Records are band-passed 1-45 Hz (after demean and a 5% taper) as for RED-PAN.

Models (``--model``): phasenet_stead phasenet_instance eqt_stead eqt_instance
Datasets: stead geonet crew tw instance romplus (Table III; earthquakes + the dataset's noise),
          stead_noise_test, geonet_noise_test, rocknet_noise_test (noise only; Fig. 3 pooled noise).
The published PN-S / EQT-S STEAD and TW rows used the first 30,000 earthquakes (``--max-eq 30000``).
A record whose processing raises is logged and skipped, as in the manuscript run; the run exits
with an error if no record was read or every record failed.

Example:
    python run_native.py --model eqt_stead --dataset crew
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from rpm_bench import config, datasets, models
from rpm_bench.native import load_model, process_trace

MODELS = {
    "phasenet_stead": "phasenet:stead",
    "phasenet_instance": "phasenet:instance",
    "eqt_stead": "eqt:stead",
    "eqt_instance": "eqt:instance",
}
H5_DIRS = {"crew": "CREW", "tw": "TW", "instance": "INSTANCE", "romplus": "ROMPLUS"}
DATASETS = (
    ("stead", "geonet")
    + tuple(H5_DIRS)
    + ("stead_noise_test", "geonet_noise_test", "rocknet_noise_test")
)


def source(ds: str, cfg) -> tuple[Path, str]:
    """(location, config key) a dataset is read from, for error messages."""
    if ds in H5_DIRS:
        return cfg["h5_root"] / H5_DIRS[ds], "h5_root"
    if ds in ("geonet_noise_test", "rocknet_noise_test"):
        return cfg["h5_root"], "h5_root"
    key = {
        "stead": "stead_root",
        "geonet": "geonet_holdout_root",
        "stead_noise_test": "stead_noise_test_dir",
    }[ds]
    return cfg[key], key


def iterators(ds, cfg, max_eq, max_noise):
    """(earthquake iterator, noise iterator) of a dataset, band-passed 1-45 Hz."""
    f = "bp145"
    if ds == "stead":
        mc, mh = cfg["stead_root"] / "merge.csv", cfg["stead_root"] / "merge.hdf5"
        return (
            datasets.iter_stead_eq(mc, mh, max_eq, test_only=True, filt=f),
            datasets.iter_stead_noise(mc, mh, max_noise, test_only=True, filt=f),
        )
    if ds == "geonet":
        r = cfg["geonet_holdout_root"]
        return datasets.iter_geonet_eq(r, max_eq, filt=f), datasets.iter_geonet_noise(
            r, max_noise, filt=f
        )
    if ds in H5_DIRS:
        r = cfg["h5_root"] / H5_DIRS[ds]
        return (
            datasets.iter_h5_dataset(r, "eq", max_eq, filt=f),
            datasets.iter_h5_dataset(r, "noise", max_noise, filt=f),
        )
    if (
        ds == "stead_noise_test"
    ):  # the original 60 s trace, as the baselines read it from merge.hdf5
        return iter(()), datasets.iter_stead_noise_test(
            cfg["stead_noise_test_dir"], 3000, max_noise, filt=f
        )
    pool = {"geonet_noise_test": "GeoNet", "rocknet_noise_test": "RockNet"}[ds]
    return iter(()), datasets.iter_h5_noise_pool(cfg["h5_root"], pool, max_noise, filt=f)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument("--model", required=True, choices=tuple(MODELS))
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--max-eq", type=int, default=0, help="cap on earthquake records (0 = all)")
    p.add_argument("--max-noise", type=int, default=0, help="cap on noise records (0 = all)")
    p.add_argument(
        "--out", type=Path, default=None, help="default <results_root>/native/<model>/<dataset>.csv"
    )
    p.add_argument(
        "--device",
        default=models.default_device(),
        help="torch device (default: cuda:0 if available, else cpu)",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config.load(args.config)
    model, model_name = load_model(MODELS[args.model], args.device)
    eq_it, nz_it = iterators(args.dataset, cfg, args.max_eq, args.max_noise)
    out = args.out or config.results_dir(cfg, "native", args.model) / f"{args.dataset}.csv"
    rows, t0 = [], time.time()
    n_seen = n_fail = 0
    for wf, info in (x for it in (eq_it, nz_it) for x in it):
        n_seen += 1
        try:
            rows.append(process_trace(model_name, model, wf, info, args.device))
        except Exception as e:
            # as in the manuscript run: a failing record is logged and skipped
            n_fail += 1
            if n_fail == 1:
                logging.exception(
                    "record %s failed (first failure; traceback follows)", info["evid"]
                )
            else:
                logging.error("record %s failed: %s", info["evid"], e)
        if n_seen % 500 == 0:
            logging.info("  %d records (%.1fs)", n_seen, time.time() - t0)
    if n_seen == 0:
        raise config.no_records_error(*source(args.dataset, cfg))
    if n_fail == n_seen:
        raise SystemExit(f"all {n_seen} records failed; see the first traceback above")
    if n_fail:
        logging.warning("%d of %d records failed and were skipped", n_fail, n_seen)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    logging.info("wrote %d rows -> %s (%.1fs)", len(rows), out, time.time() - t0)


if __name__ == "__main__":
    main()
