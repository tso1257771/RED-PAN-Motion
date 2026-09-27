"""P-pick TIMING: does the inference band-pass shift the P arrival vs raw / highpass?

Tests the claim that the P pick must be made on the raw waveform. For each CEED event
the picker is run on THREE inputs and its P-probability peak near the catalogue P is
located; we report pick MAE + signed bias + the peak P-prob (detection confidence).

    band-pass : demean + 3-45 Hz (an inference-only preprocessing)
    raw       : raw counts, z-scored only (what CEED was actually TRAINED on)
    highpass  : 1.0 Hz highpass (what TW / GeoNet were trained on)

CONFOUND: training applied per-dataset HP only to {TW:1.0, GeoNet:1.0}; CEED trained
on RAW (no filter). So for CEED, 'raw' is in-distribution and band-pass/highpass are
inference-time mismatches — a raw win here is partly distribution-match, not proof raw
is universally better. Read alongside that caveat.

    PYTHONPATH=. python scripts/benchmarks/benchmark_p_pick_timing.py --max-eq 2500
"""
import argparse
import os
from pathlib import Path

import numpy as np
import h5py
import torch
from scipy.signal import butter, sosfiltfilt

from redpan_motion import REDPANPredictor, bandpass_for_model

CACHE = os.environ.get("CEED_CACHE", "/path/to/data/seisbench_cache/datasets/ceed")
# CEED_TEST_IDS: a .npy of the CEED test-split trace ids to evaluate on.
IDS = os.environ.get("CEED_TEST_IDS", "/path/to/ceed_test_ids.npy")
CKPT = str(Path(__file__).resolve().parents[2] / "checkpoints/redpan_motion/best.pt")
IN_SAMPLES = 9000
DT = 0.01


def highpass_for_model(raw, f, dt=DT):
    """Demean + 4th-order zero-phase Butterworth highpass (mirrors bandpass_for_model)."""
    nyq = 0.5 / dt
    sos = butter(4, f / nyq, btype="highpass", output="sos")
    out = np.empty(raw.shape, np.float32)
    for c in range(raw.shape[0]):
        out[c] = sosfiltfilt(sos, raw[c] - raw[c].mean()).astype(np.float32)
    return out


def crop_to_window(wf, p_sample, target_len=IN_SAMPLES):
    n = wf.shape[1]
    start = p_sample - target_len // 2
    out = np.zeros((3, target_len), np.float32)
    lo, hi = max(0, start), min(n, start + target_len)
    out[:, lo - start:lo - start + (hi - lo)] = wf[:, lo:hi]
    return out, p_sample - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eq", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--win", type=float, default=0.5, help="± search window (s) around catalogue P")
    ap.add_argument("--min-prob", type=float, default=0.1, help="P-prob below this = miss")
    args = ap.parse_args()

    ids = np.load(IDS, allow_pickle=False)
    ids = ids[np.random.default_rng(args.seed).permutation(len(ids))][:args.max_eq]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = REDPANPredictor.from_checkpoint(CKPT, device=device)

    h5c = {}

    def get_h5(region, year):
        key = (region, year)
        if key not in h5c:
            p = Path(CACHE) / f"waveforms{region}{year}.hdf5"
            h5c[key] = h5py.File(p, "r") if p.exists() else None
        return h5c[key]

    inputs = {
        "band-pass": lambda w: bandpass_for_model(w, DT),
        "raw": lambda w: w,
        "highpass1.0": lambda w: highpass_for_model(w, 1.0),
    }
    W = int(args.win / DT)
    rows = []
    for r in ids:
        tn = str(r["trace_name"])
        h5 = get_h5(str(r["region"]), int(r["year"]))
        if h5 is None or tn not in h5:
            continue
        wf = h5[tn][:]
        if wf.shape[0] != 3:
            continue
        win, p = crop_to_window(wf.astype(np.float32), int(r["p_sample"]))
        lo, hi = max(0, p - W), min(IN_SAMPLES, p + W)
        row = {}
        for k, fn in inputs.items():
            picker, _, _ = pred.predict_arrays(fn(win), mode="single")  # (T,3) [P,S,N]
            seg = picker[lo:hi, 0]
            pv = float(seg.max())
            row[k] = ((lo + int(seg.argmax()) - p) * DT, pv) if pv >= args.min_prob else None
        rows.append(row)

    def report(subset, title):
        print(f"\n{title}  (n={len(subset)})")
        print(f"  {'input':12s} | picks | MAE(s) | bias(s) | median(s) | %<0.05s | %<0.1s | meanP")
        for k in inputs:
            v = [row[k] for row in subset if row[k]]
            if not v:
                print(f"  {k:12s} | 0"); continue
            res = np.array([x[0] for x in v]); pr = np.array([x[1] for x in v])
            print(f"  {k:12s} | {len(res):5d} | {np.abs(res).mean():.3f}  | {res.mean():+.3f}  | "
                  f"{np.median(res):+.3f}    | {(np.abs(res) < 0.05).mean():.2f}    | "
                  f"{(np.abs(res) < 0.1).mean():.2f}   | {pr.mean():.2f}")

    print(f"P-pick timing vs catalogue P — shuffled CEED, ±{args.win}s search window")
    print("NOTE: CEED trained RAW (no per-dataset HP); band-pass 3-45 is inference-only. See caveat in header.")
    report(rows, "Per-input (each input's own confident picks)")
    paired = [row for row in rows if all(row[k] for k in inputs)]
    report(paired, "Paired (events where ALL three inputs pick confidently)")


if __name__ == "__main__":
    main()
