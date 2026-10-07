#!/usr/bin/env python
"""CEED first-motion polarity (Table V): per-record outputs of one 90 s model.

Records: the 99,998 CEED test traces listed in ``ceed_test_ids`` (NC + SC, 2021-2023) with a
``trace_p_polarity`` label (U / D / N) in the SeisBench CEED metadata; waveforms from the SeisBench
CEED cache (``ceed_cache``: metadata*.csv, waveforms<region><year>.hdf5).
Writes ``<results_root>/polarity/<model>_polarity_per_event.csv`` (gt_polarity, pred_polarity,
polarity_u_prob, polarity_d_prob, ..., one row per record). Score it with ``score_table5.py``.
Records whose waveform file or trace is missing are skipped and counted in the log; the run exits
with an error if no record was read.

Example:
    python run_polarity.py --model edge
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rpm_bench import config, models
from rpm_bench.constants import DT, SENTINEL
from rpm_bench.polarity import (
    build_polarity_index,
    crop_to_window,
    decode_polarity,
    detection_picking,
    open_year_files,
)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument("--model", required=True, choices=("edge", "rpm"))
    p.add_argument("--in-samples", type=int, default=9000)
    p.add_argument(
        "--imp-threshold",
        type=float,
        default=0.5,
        help="only used by 2-channel heads (not edge/rpm)",
    )
    p.add_argument(
        "--max-eq",
        "--max-records",
        dest="max_eq",
        type=int,
        default=0,
        help="cap on records (0 = all)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default <results_root>/polarity/<model>_polarity_per_event.csv",
    )
    p.add_argument(
        "--device",
        default=models.default_device(),
        help="torch device (default: cuda:0 if available, else cpu)",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(asctime)s : %(message)s")
    cfg = config.load(args.config)

    ids = np.load(cfg["ceed_test_ids"], allow_pickle=False)
    pol_idx = build_polarity_index(cfg["ceed_cache"])
    ids_keep = ids[np.array([pol_idx.get(str(r["trace_name"])) in ("U", "D", "N") for r in ids])]
    if args.max_eq > 0:
        ids_keep = ids_keep[: args.max_eq]
    logging.info("records with a polarity label: %d / %d", len(ids_keep), len(ids))
    regions_years = sorted({(str(r["region"]), int(r["year"])) for r in ids_keep})
    h5_handles = open_year_files(cfg["ceed_cache"], regions_years)
    missing = [f"waveforms{r}{y}.hdf5" for r, y in regions_years if (r, y) not in h5_handles]
    if missing:
        logging.warning(
            "%d of %d waveform files not found in %s: %s",
            len(missing),
            len(regions_years),
            cfg["ceed_cache"],
            ", ".join(missing),
        )

    inner = models.load_model(args.model, cfg, args.device).backbone.inner
    pol_head = getattr(inner, "polarity_head", None) or getattr(inner, "polarity", None)
    oc = int(
        getattr(getattr(pol_head, "weight", None), "shape", [inner.polarity_output_channels])[0]
    )
    pol_mode = "softmax_ce" if oc == 3 else "softmax_ud" if oc == 2 else "bce"
    logging.info("polarity output mode: %s (out_ch=%d)", pol_mode, oc)

    results, t0 = [], time.time()
    n_no_file = n_no_trace = n_not_3c = 0
    for row in ids_keep:
        trace_name = str(row["trace_name"])
        region = str(row["region"])
        year = int(row["year"])
        p_sample = int(row["p_sample"])
        s_sample = int(row["s_sample"])
        gt_pol = pol_idx[trace_name]
        h5 = h5_handles.get((region, year))
        if h5 is None:
            n_no_file += 1
            continue
        if trace_name not in h5:
            n_no_trace += 1
            continue
        wf = h5[trace_name][:]
        if wf.shape[0] != 3:
            n_not_3c += 1
            continue
        cropped, new_p, _ = crop_to_window(wf.astype(np.float32), p_sample, args.in_samples)
        z_raw = cropped[2:3].copy()
        z_max = float(np.max(np.abs(z_raw)))
        if z_max > 1e-6:
            z_raw = z_raw / z_max
        mean = cropped.mean(axis=1, keepdims=True)
        std = cropped.std(axis=1, keepdims=True)
        x_norm = (cropped - mean) / np.where(std > 1e-8, std, 1.0)
        x = torch.from_numpy(x_norm[None]).to(args.device).float()
        z_raw_t = torch.from_numpy(z_raw[None]).to(args.device).float()
        with torch.no_grad():
            out = inner(x, z_raw=z_raw_t)
        if len(out) == 4:
            picker, polarity, impulsive, detector = out
        else:
            picker, polarity, detector = out
            impulsive = None
        imp_score = float(impulsive[0, 0, new_p].item()) if impulsive is not None else float("nan")
        pred, signed, u_prob, d_prob = decode_polarity(
            polarity[0], new_p, imp_score, args.imp_threshold, pol_mode
        )
        p_prob = float(picker[0, 0, new_p].item())
        s_offset = s_sample - p_sample + new_p
        s_prob = (
            float(picker[0, 1, s_offset].item())
            if 0 <= s_offset < args.in_samples
            else float("nan")
        )
        det_at_p = float(detector[0, 0, new_p].item())
        det_pick = detection_picking(picker, detector, new_p, s_offset, args.in_samples)
        if det_pick["own_P_idx"] >= 0:
            pred_at_pick, signed_at_pick, _, _ = decode_polarity(
                polarity[0],
                det_pick["own_P_idx"],
                float(impulsive[0, 0, det_pick["own_P_idx"]].item())
                if impulsive is not None
                else float("nan"),
                args.imp_threshold,
                pol_mode,
            )
        else:
            pred_at_pick = "N"
        own_p_sec = det_pick["own_P_idx"] * DT if det_pick["own_P_idx"] >= 0 else float("nan")
        own_s_sec = det_pick["own_S_idx"] * DT if det_pick["own_S_idx"] >= 0 else float("nan")
        label_p_sec = new_p * DT
        label_s_sec = s_offset * DT if 0 <= s_offset < args.in_samples else float("nan")
        results.append(
            {
                "trace_name": trace_name,
                "region": region,
                "year": year,
                "magnitude": float(row["magnitude"]),
                "distance_km": float(row["distance_km"]),
                "p_sample": p_sample,
                "s_sample": s_sample,
                "gt_polarity": gt_pol,
                "pred_polarity": pred,
                "pred_polarity_at_own_pick": pred_at_pick,
                "polarity_signed": signed,
                "polarity_u_prob": u_prob,
                "polarity_d_prob": d_prob,
                "impulsive_score": imp_score,
                "P_prob_at_label": p_prob,
                "S_prob_at_label": s_prob,
                "det_prob_at_label": det_at_p,
                "n_triggers": det_pick["n_triggers"],
                "trigger_on_sec": det_pick["trigger_on"] * DT
                if det_pick["trigger_on"] >= 0
                else SENTINEL,
                "trigger_off_sec": det_pick["trigger_off"] * DT
                if det_pick["trigger_off"] >= 0
                else SENTINEL,
                "mask_peak_in_trigger": det_pick["mask_peak"],
                "mask_mean_in_trigger": det_pick["mask_mean"],
                "own_P_pick_sec": own_p_sec,
                "own_P_prob": det_pick["own_P_prob"],
                "own_S_pick_sec": own_s_sec,
                "own_S_prob": det_pick["own_S_prob"],
                "P_residual_sec": (own_p_sec - label_p_sec)
                if not np.isnan(own_p_sec)
                else float("nan"),
                "S_residual_sec": (own_s_sec - label_s_sec)
                if (not np.isnan(own_s_sec) and not np.isnan(label_s_sec))
                else float("nan"),
            }
        )
        if len(results) % 5000 == 0:
            logging.info("  %d / %d (%.1fs)", len(results), len(ids_keep), time.time() - t0)
    for h in h5_handles.values():
        h.close()
    n_skip = n_no_file + n_no_trace + n_not_3c
    if n_skip:
        logging.warning(
            "skipped %d of %d records: %d in a missing waveform file, %d not in their file, "
            "%d not 3-component",
            n_skip,
            len(ids_keep),
            n_no_file,
            n_no_trace,
            n_not_3c,
        )
    if not results:
        raise SystemExit(
            f"no records read from {cfg['ceed_cache']}; check config keys ceed_cache / "
            f"RPM_BENCH_CEED_CACHE and ceed_test_ids / RPM_BENCH_CEED_TEST_IDS"
        )
    out = args.out or config.results_dir(cfg, "polarity") / f"{args.model}_polarity_per_event.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    logging.info("wrote %d records -> %s (%.1fs)", len(results), out, time.time() - t0)


if __name__ == "__main__":
    main()
