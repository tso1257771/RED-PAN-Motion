#!/usr/bin/env python
"""Static benchmark: one window per record, every mask trigger written as a CSV row.

Writes ``<results_root>/static/<model>/<dataset>.csv`` with one row per (record, trigger), plus
one row per untriggered record (trigger fields = -999):

    evid, label_type, mode, model, labelP_sec, labelS_sec, ps_diff_sec, polarity_label,
    n_triggers, trigger_idx, trigger_on_sec, trigger_off_sec, mask_peak, mask_mean,
    P_pick_sec, P_pick_prob, S_pick_sec, S_pick_prob, polarity_at_P, P_residual_sec, S_residual_sec

Earthquake records: the window of the model's length is cut with the labeled P at 10% of it
(6 s for the 60 s RED-PAN, 9 s for the 90 s models), padded with spectrum-matched noise where it
leaves the record. Noise records: centre crop (or the original 60 s for STEAD test noise and the
60 s models). Triggers: mask smoothed over 10 samples, ``trigger_onset`` 0.1/0.1.

Datasets (``--dataset``):
  ceed_nc ceed_sc crew instance romplus tw   90 s HDF5 test splits (earthquakes; INSTANCE/TW also noise)
  stead_h5                                   STEAD HDF5 test split (Table II, IV)
  geonet_val                                 GeoNet HDF5 validation split (not in the manuscript)
  stead                                      original STEAD test.npy earthquakes + noise (Table III, Fig. 3)
  geonet                                     GeoNet 2013-2014 holdout, earthquakes + noise (Table II-IV, Fig. 3)
  stead_noise_test                           STEAD official test noise only (Table III STEAD noise)

Large splits can be run in chunks with ``--rows start:stop``; the slice applies to the earthquake
records only (the noise iterator reads the whole noise split), so pass ``--no-noise`` to every
chunk but one and combine the CSVs with ``pandas.concat`` (not ``cat``, which repeats the header).

Example:
    python run_static.py --model edge --dataset crew
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rpm_bench import config, datasets, models
from rpm_bench.constants import DT
from rpm_bench.static import build_rows, extract_triggers, run_static

H5_DIRS = {
    "ceed_nc": "CEED_NC",
    "ceed_sc": "CEED_SC",
    "crew": "CREW",
    "instance": "INSTANCE",
    "romplus": "ROMPLUS",
    "tw": "TW",
    "stead_h5": "STEAD",
    "geonet_val": "GeoNet",
}
DATASETS = tuple(H5_DIRS) + ("stead", "geonet", "stead_noise_test")
NO_NOISE_BY_DEFAULT = {
    "stead_h5",
    "geonet_val",
}  # their HDF5 noise is a validation split; pooled noise is scored separately


def source(ds: str, cfg) -> tuple[Path, str]:
    """(location, config key) a dataset is read from, for error messages."""
    if ds in H5_DIRS:
        return cfg["h5_root"] / H5_DIRS[ds], "h5_root"
    key = {
        "stead": "stead_root",
        "geonet": "geonet_holdout_root",
        "stead_noise_test": "stead_noise_test_dir",
    }[ds]
    return cfg[key], key


def iterators(
    ds: str, cfg, filt: str, is60: bool, max_eq: int, max_noise: int, with_noise: bool, rows
):
    """(earthquake iterator, noise iterator) of a dataset; see rpm_bench.datasets."""
    if ds in H5_DIRS:
        root = cfg["h5_root"] / H5_DIRS[ds]
        eq = datasets.iter_h5_dataset(root, "eq", max_eq, filt=filt, rows=rows)
        nz = (
            datasets.iter_h5_dataset(root, "noise", max_noise, filt=filt)
            if with_noise
            else iter(())
        )
    elif ds == "stead":
        mc, mh = cfg["stead_root"] / "merge.csv", cfg["stead_root"] / "merge.hdf5"
        eq = datasets.iter_stead_eq(mc, mh, max_eq, test_only=True, filt=filt)
        nz = (
            datasets.iter_stead_noise(mc, mh, max_noise, test_only=True, filt=filt)
            if with_noise
            else iter(())
        )
    elif ds == "geonet":
        eq = datasets.iter_geonet_eq(cfg["geonet_holdout_root"], max_eq, filt=filt)
        nz = (
            datasets.iter_geonet_noise(cfg["geonet_holdout_root"], max_noise, filt=filt)
            if with_noise
            else iter(())
        )
    elif ds == "stead_noise_test":  # 60 s models read the original 60 s trace (samples 3000:9000)
        eq = iter(())
        nz = datasets.iter_stead_noise_test(
            cfg["stead_noise_test_dir"], 3000 if is60 else 0, max_noise, filt=filt
        )
    else:
        raise ValueError(ds)
    return eq, nz


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument("--model", required=True, choices=models.MODEL_KEYS)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument(
        "--filter",
        default=None,
        choices=["bp145", "hp1", "raw"],
        help="input filter (default: the model's benchmark filter, hp1 for the 90 s models, bp145 for RED-PAN)",
    )
    p.add_argument("--max-eq", type=int, default=0, help="cap on earthquake records (0 = all)")
    p.add_argument("--max-noise", type=int, default=0, help="cap on noise records (0 = all)")
    p.add_argument("--no-noise", action="store_true", help="skip the dataset's noise records")
    p.add_argument(
        "--rows",
        default=None,
        help="'start:stop' slice of the HDF5 earthquake split, to run one chunk in parallel "
        "(noise is not sliced: use --no-noise on all chunks but one)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output CSV (default <results_root>/static/<model>/<dataset>.csv)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed numpy's global RNG, which draws the spectrum-matched padding "
        "(default: unseeded, as in the manuscript runs)",
    )
    p.add_argument(
        "--device",
        default=models.default_device(),
        help="torch device (default: cuda:0 if available, else cpu)",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.seed is not None:
        np.random.seed(args.seed)

    cfg = config.load(args.config)
    filt = args.filter or models.FILTER[args.model]
    model = models.load_model(args.model, cfg, args.device)
    in_samples = model.in_samples
    with_noise = not (args.no_noise or args.dataset in NO_NOISE_BY_DEFAULT)
    eq_it, nz_it = iterators(
        args.dataset,
        cfg,
        filt,
        model.is_redpan60,
        args.max_eq,
        args.max_noise,
        with_noise,
        args.rows,
    )
    out = args.out or config.results_dir(cfg, "static", args.model) / f"{args.dataset}.csv"
    logging.info(
        "model=%s dataset=%s filter=%s in_samples=%d -> %s",
        args.model,
        args.dataset,
        filt,
        in_samples,
        out,
    )

    rows: list[dict] = []
    t0 = time.time()
    n_done = 0
    for wf, info in (x for it in (eq_it, nz_it) for x in it):
        p_abs = int(round(info["labelP_sec"] / DT)) if info["labelP_sec"] >= 0 else None
        s_abs = int(round(info["labelS_sec"] / DT)) if info["labelS_sec"] >= 0 else None
        mask, p_arr, s_arr, pol = run_static(
            model, wf, in_samples, args.device, p_abs=p_abs, s_abs=s_abs
        )
        triggers = extract_triggers(
            mask, p_arr, s_arr, pol, dt=DT, thr_on=0.1, thr_off=0.1, smooth_npts=10
        )
        rec = dict(
            evid=info["evid"],
            label_type=info["label_type"],
            mode="static",
            model=models.LABEL[args.model],
        )
        rows.extend(
            build_rows(
                rec,
                triggers,
                info["labelP_sec"],
                info["labelS_sec"],
                info.get("polarity_label", ""),
            )
        )
        n_done += 1
        if n_done % 1000 == 0:
            logging.info("  %d records (%.1fs)", n_done, time.time() - t0)
    if n_done == 0:
        raise config.no_records_error(*source(args.dataset, cfg))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    logging.info(
        "wrote %d rows for %d records -> %s (%.1fs)", len(rows), n_done, out, time.time() - t0
    )


if __name__ == "__main__":
    main()
