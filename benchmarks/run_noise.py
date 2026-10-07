#!/usr/bin/env python
"""Pooled test noise (Tables II and IV, Fig. 3): one model on one noise pool.

The pooled noise of the manuscript is 92,213 records in five pools:

    STEAD_test     23,526  official STEAD test noise (build_stead_noise_test.py)
    GeoNet_test    16,384  90 s HDF5 noise, test split
    INSTANCE_test  19,838
    RockNet_test    1,593
    TW_test        30,872

``--pool STEAD_val`` reads the validation split of the 90 s archive's own STEAD noise instead
(noise traces outside STEAD test.npy; see build_stead_noise_test.py). No scorer reads it.

Writes, in ``<results_root>/noise/<model>/``:
  noisefp_<POOL>.csv           one row per record: whole-record peaks and the detector-trigger
                               columns (det_trigger_max: max mask over trigger_onset 0.3/0.3 triggers)
  noisefp_triggers_<POOL>.csv  one row per mask trigger (smoothed mask, onset 0.1, the same recipe as
                               run_static.py): mask_peak, mask_mean, inside_max_P, inside_max_S

Input: the model's filter (hp1 for the 90 s models, bp145 for RED-PAN) applied zero-phase to the record,
then a per-channel z-score. The 60 s RED-PAN reads the first 6,000 samples of a pool record, or
samples 3000:9000 (the original 60 s trace) of the STEAD test noise.
This differs from run_static.py on purpose (it is the manuscript's noise run): no demean or 5%
Tukey taper before the filter, and the z-score divides by ``std`` where ``std > 1e-8`` (else by 1)
instead of by ``std + 1e-10``. No random padding is involved, so a re-run is deterministic.

Example:
    python run_noise.py --model edge --pool TW_test
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from obspy.signal.trigger import trigger_onset
from rpm_bench import config, models
from rpm_bench.constants import DT, SENTINEL
from rpm_bench.static import extract_triggers
from scipy.signal import butter, find_peaks, sosfiltfilt

POOLS = ("STEAD_test", "GeoNet_test", "INSTANCE_test", "RockNet_test", "TW_test", "STEAD_val")
REDPAN60S_CROP = 6000


def pool_files(pool: str, cfg):
    """(metadata csv, h5, split, 60 s crop start) of a pool."""
    if pool == "STEAD_test":
        d = cfg["stead_noise_test_dir"]
        return (
            d / "STEAD_dataset_90s_noise_test_metadata.csv",
            d / "STEAD_dataset_90s_noise_test.h5",
            "test",
            3000,
        )
    name, split = pool.split("_")
    d = cfg["h5_root"] / name
    return (
        d / f"{name}_dataset_90s_noise_metadata.csv",
        d / f"{name}_dataset_90s_noise.h5",
        split,
        0,
    )


def apply_filter(wfs, filt):
    """Zero-phase filter (B, 3, T) at 100 Hz for the picker/detector input (z_raw stays raw)."""
    if filt == "raw":
        return wfs
    nyq = 50.0
    if filt == "hp1":
        sos = butter(4, 1.0 / nyq, btype="high", output="sos")
    elif filt == "bp145":
        sos = butter(4, [1.0 / nyq, 45.0 / nyq], btype="band", output="sos")
    else:
        raise ValueError(filt)
    return sosfiltfilt(sos, wfs, axis=2).astype(np.float32)


def build_trigger_rows(idx: int, cat: str, mask_arr, p_arr, s_arr) -> list:
    """One row per trigger; a record without triggers gets one trigger_idx = -1 row.
    inside_max_P / inside_max_S are extract_triggers' P_pick_prob / S_pick_prob (max inside the trigger)."""
    triggers = extract_triggers(
        mask_arr, p_arr, s_arr, polarity=None, dt=DT, thr_on=0.1, thr_off=0.1, smooth_npts=10
    )
    n = len(triggers)
    if n == 0:
        return [
            {
                "hdf5_index": idx,
                "category": cat,
                "n_triggers": 0,
                "trigger_idx": -1,
                "trigger_on_sec": SENTINEL,
                "trigger_off_sec": SENTINEL,
                "mask_peak": 0.0,
                "mask_mean": 0.0,
                "inside_max_P": 0.0,
                "inside_max_S": 0.0,
            }
        ]
    return [
        {
            "hdf5_index": idx,
            "category": cat,
            "n_triggers": n,
            "trigger_idx": t["trigger_idx"],
            "trigger_on_sec": t["trigger_on_sec"],
            "trigger_off_sec": t["trigger_off_sec"],
            "mask_peak": t["mask_peak"],
            "mask_mean": t["mask_mean"],
            "inside_max_P": t["P_pick_prob"],
            "inside_max_S": t["S_pick_prob"],
        }
        for t in triggers
    ]


def _channels_first(w: np.ndarray) -> np.ndarray:
    """(3, T) as stored, or the transpose of a (T, 3) record."""
    return w if w.shape[0] == 3 else w.T


def find_noise_group(h5_handle) -> str:
    """Top-level group of the noise waveforms in a pool HDF5 ("" if the splits are at the top)."""
    for c in ("noise", "Noise", "noise/noise"):
        if c in h5_handle:
            return c
    if any(k in h5_handle for k in ("train", "val", "test")):
        return ""
    return list(h5_handle.keys())[0] if h5_handle.keys() else "noise"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument("--model", required=True, choices=models.MODEL_KEYS)
    p.add_argument("--pool", required=True, choices=POOLS)
    p.add_argument(
        "--filter",
        default=None,
        choices=["bp145", "hp1", "raw"],
        help="default: the model's filter",
    )
    p.add_argument(
        "--max-records",
        "--max-noise",
        dest="max_records",
        type=int,
        default=0,
        help="cap on noise records (0 = all)",
    )
    p.add_argument("--batch-size", type=int, default=64, help="records per forward pass")
    p.add_argument(
        "--peak-min", type=float, default=0.01, help="min peak height kept in the per-record CSV"
    )
    p.add_argument(
        "--out-dir", type=Path, default=None, help="default <results_root>/noise/<model>"
    )
    p.add_argument(
        "--device",
        default=models.default_device(),
        help="torch device (default: cuda:0 if available, else cpu)",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = config.load(args.config)
    filt = args.filter or models.FILTER[args.model]
    csv, h5p, split, crop60 = pool_files(args.pool, cfg)
    df = pd.read_csv(csv, low_memory=False)
    df = df[df["split"] == split].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(
            f"no '{split}' records in {csv}; check the pool and config key "
            f"{'stead_noise_test_dir' if args.pool == 'STEAD_test' else 'h5_root'}"
        )
    if args.max_records > 0:
        df = df.iloc[: args.max_records].reset_index(drop=True)
    out_dir = args.out_dir or config.results_dir(cfg, "noise", args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv, trig_csv = (
        out_dir / f"noisefp_{args.pool}.csv",
        out_dir / f"noisefp_triggers_{args.pool}.csv",
    )

    model = models.load_model(args.model, cfg, args.device)
    inner = model.backbone.inner
    is60 = model.is_redpan60
    # As in the manuscript run: z_raw is passed only to networks with a `polarity_head`
    # (RED-PAN-Motion); Edge-RED-PAN-Motion's picker/detector do not read it.
    use_zraw = hasattr(inner, "polarity_head")
    rows, trig_rows = [], []
    meta = [
        dict(cat=str(r.get("category", "noise")), idx=int(r["hdf5_index"]))
        for _, r in df.iterrows()
    ]
    t0 = time.time()
    with h5py.File(h5p, "r") as h5:
        top = find_noise_group(h5)
        key = f"{top}/{split}/waveforms" if top else f"{split}/waveforms"
        logging.info(
            "model=%s pool=%s (%d records) filter=%s key=%s",
            args.model,
            args.pool,
            len(df),
            filt,
            key,
        )
        for b in range(0, len(meta), args.batch_size):
            batch = meta[b : b + args.batch_size]
            wfs = np.stack([_channels_first(h5[key][m["idx"]]).astype(np.float32) for m in batch])
            win = wfs[:, :, crop60 : crop60 + REDPAN60S_CROP] if is60 else wfs
            x = apply_filter(win, filt)
            sd = x.std(axis=2, keepdims=True)
            x = (x - x.mean(axis=2, keepdims=True)) / np.where(sd > 1e-8, sd, 1.0)
            x_t = torch.from_numpy(x).to(args.device).float()
            with torch.no_grad():
                if use_zraw:
                    z = wfs[:, 2:3, :].copy()
                    zm = np.max(np.abs(z), axis=2, keepdims=True)
                    z = z / np.where(zm > 1e-6, zm, 1.0)
                    out = inner(x_t, z_raw=torch.from_numpy(z).to(args.device).float())
                else:
                    out = inner(x_t)
            picker, detector = out[0], out[-1]
            Pa, Sa, Da = (
                picker[:, 0].cpu().numpy(),
                picker[:, 1].cpu().numpy(),
                detector[:, 0].cpu().numpy(),
            )
            for k, m in enumerate(batch):
                P, S, D = Pa[k], Sa[k], Da[k]
                pp, ppr = find_peaks(P, height=args.peak_min, distance=100)
                sp, spr = find_peaks(S, height=args.peak_min, distance=100)
                pv = ppr["peak_heights"] if len(pp) else np.array([])
                sv = spr["peak_heights"] if len(sp) else np.array([])
                trigs = trigger_onset(D, 0.3, 0.3)
                # hi + 1: unlike the [lo:hi] slices of extract_triggers, this includes trigger_onset's
                # inclusive end sample (README, Known caveats)
                det_trig_max = (
                    float(max(D[lo : hi + 1].max() for lo, hi in trigs)) if len(trigs) else 0.0
                )
                rows.append(
                    {
                        "hdf5_index": m["idx"],
                        "category": m["cat"],
                        "n_P_peaks_above_min": len(pp),
                        "n_S_peaks_above_min": len(sp),
                        "P_peak_values": ";".join(f"{v:.4f}" for v in pv),
                        "S_peak_values": ";".join(f"{v:.4f}" for v in sv),
                        "max_P": float(pv.max()) if len(pv) else 0.0,
                        "max_S": float(sv.max()) if len(sv) else 0.0,
                        "det_max": float(D.max()),
                        "det_mean": float(D.mean()),
                        "n_det_triggers": len(trigs),
                        "det_trigger_max": det_trig_max,
                    }
                )
                trig_rows.extend(build_trigger_rows(m["idx"], m["cat"], D, P, S))
            if (b // args.batch_size) % 50 == 0:
                logging.info("  %d/%d (%.1fs)", b + len(batch), len(meta), time.time() - t0)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(trig_rows).to_csv(trig_csv, index=False)
    logging.info("wrote %s and %s (%.1fs)", out_csv, trig_csv, time.time() - t0)


if __name__ == "__main__":
    main()
