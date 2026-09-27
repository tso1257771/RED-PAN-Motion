"""Benchmark redpan_motion picking on the STEAD test set — single forward pass.

No SeisBench wrapper: inference goes straight through `REDPANPredictor`, ONE
forward pass per trace (the single-forward-pass mode). STEAD traces are 60 s
(6000 samples) < the model's 90 s window, so each is embedded in a 90 s window:
  * `--pad noise` (default): EQ placed with P at 10%, spectrum-matched noise
    front/back pad.
  * `--pad zero`: `predict_array` short-path zero-pads to 90 s (simpler).
Noise traces always use the short-path zero-pad.

Metrics (trigger-based, RED-PAN convention):
  * per EQ trace, the strongest detector trigger gives the P/S pick; a pick is
    correct if within `--tol` s of the catalogue label.
  * P/S recall, residual MAE / MAD / std (on matched picks)
  * noise false-positive rate = fraction of NOISE traces with any trigger
  * precision / F1 (FP = mislocated EQ picks + noise picks)

Usage:
    PYTHONPATH=. python scripts/benchmarks/benchmark_stead.py --max-eq 2000 --max-noise 2000
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
from obspy.signal.trigger import trigger_onset
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

from redpan_motion.inference import REDPANPredictor
from redpan_motion.utils.waveform import generate_matching_noise

DT = 0.01
TRIGGER_THR = 0.1           # detector on/off threshold for trigger_onset
TRIGGER_SMOOTH_NPTS = 10    # boxcar width to smooth the mask before triggering
STEAD_DIR = Path(os.environ.get("STEAD_DIR", "/path/to/data/STEAD"))
STEAD_CSV = STEAD_DIR / "merge.csv"
STEAD_HDF5 = STEAD_DIR / "merge.hdf5"
STEAD_TEST_IDS = STEAD_DIR / "test.npy"
CKPT = "redpan_motion"   # the shipped checkpoint; or a path to a .pt


def preprocess(wf, taper_ratio=0.05, bandpass=(1.0, 45.0)):
    """(T, 3) detrend + Tukey taper + bandpass per channel (STEAD convention)."""
    wf = wf.astype(np.float32).copy()
    wf -= wf.mean(axis=0, keepdims=True)
    wf *= tukey(wf.shape[0], taper_ratio).astype(np.float32)[:, None]
    fs = 1.0 / DT
    sos = butter(4, [bandpass[0] / (0.5 * fs), bandpass[1] / (0.5 * fs)],
                 btype="band", output="sos")
    for c in range(wf.shape[1]):
        wf[:, c] = sosfiltfilt(sos, wf[:, c]).astype(np.float32)
    return wf


def build_static_window_eq(wf_TC, p_abs, s_abs, in_samples=9000,
                           p_pos_frac=0.1, coda_offset_s=2.0):
    """Embed a 60 s EQ in a 90 s window with P at `p_pos_frac` (10%); front-pad
    with spectrum-matched noise from pre-P, back-pad from post-S coda. Port of
    benchmark_unified._build_static_window_eq. Returns (win, real_start,
    real_end, front_pad_n) for mapping model output back to trace coordinates."""
    n, n_ch = wf_TC.shape
    desired_start = p_abs - int(round(p_pos_frac * in_samples))
    desired_end = desired_start + in_samples
    front_pad_n = max(0, -desired_start)
    back_pad_n = max(0, desired_end - n)
    real_start, real_end = max(0, desired_start), min(n, desired_end)
    front_pad = np.zeros((front_pad_n, n_ch), np.float32)
    if front_pad_n > 0:
        ref_pre = wf_TC[:max(0, p_abs)]
        if len(ref_pre) >= 10:
            for ch in range(n_ch):
                front_pad[:, ch] = generate_matching_noise(ref_pre[:, ch], front_pad_n).astype(np.float32)
    back_pad = np.zeros((back_pad_n, n_ch), np.float32)
    if back_pad_n > 0:
        ref_post = wf_TC[max(0, min(n - 100, s_abs + int(round(coda_offset_s / DT)))):]
        if len(ref_post) >= 10:
            for ch in range(n_ch):
                back_pad[:, ch] = generate_matching_noise(ref_post[:, ch], back_pad_n).astype(np.float32)
    win = np.concatenate([front_pad, wf_TC[real_start:real_end], back_pad], axis=0).astype(np.float32)
    assert win.shape[0] == in_samples, (win.shape, front_pad_n, back_pad_n, real_end - real_start)
    return win, real_start, real_end, front_pad_n


def smooth(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same") if n > 1 else x


def best_pick(mask, p_arr, s_arr, thr=TRIGGER_THR, smooth_npts=TRIGGER_SMOOTH_NPTS):
    """Strongest detector trigger -> (P_sec, S_sec, mask_peak) or None."""
    triggers = trigger_onset(smooth(mask, smooth_npts), thr, thr)
    if len(triggers) == 0:
        return None
    best, best_peak = None, float("-inf")
    for lo, hi in triggers:
        lo, hi = int(lo), int(hi)
        if hi <= lo:
            continue
        peak = float(mask[lo:hi].max())
        if peak > best_peak:
            best_peak = peak
            p = lo + int(p_arr[lo:hi].argmax())
            s = lo + int(s_arr[lo:hi].argmax())
            best = (p * DT, s * DT, peak)
    return best


def stats(residuals):
    """MAE / MAD / std / mean of a residual list (NaN-filled when empty)."""
    r = np.asarray(residuals, dtype=float)
    if r.size == 0:
        return dict(MAE=np.nan, MAD=np.nan, std=np.nan, mean=np.nan, n=0)
    return dict(MAE=float(np.mean(np.abs(r))), MAD=float(np.median(np.abs(r))),
                std=float(np.std(r)), mean=float(np.mean(r)), n=int(r.size))


def prf(tp, n_pos, fp):
    """Precision / recall / F1. F1 = 0.0 when precision or recall is a defined
    zero; NaN only when precision/recall are themselves undefined (no positives
    or no predictions)."""
    rec = tp / n_pos if n_pos else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    if np.isnan(prec) or np.isnan(rec):
        return prec, rec, float("nan")
    denom = prec + rec
    return prec, rec, (2 * prec * rec / denom if denom > 0 else 0.0)


def iter_stead(category, max_n, test_ids, seed=0):
    df = pd.read_csv(STEAD_CSV, low_memory=False)
    sub = df[df["trace_category"] == category]
    sub = sub[sub["trace_name"].isin(test_ids)]
    if sub.empty:
        raise RuntimeError(
            f"No '{category}' traces matched test_ids — check STEAD_CSV and "
            f"STEAD_TEST_IDS ({STEAD_TEST_IDS}).")
    # Shuffle (seeded) so a --max-N subset is REPRESENTATIVE of the full test set.
    # CSV order is grouped by station/region; the first N traces are a biased
    # slice (e.g. systematically harder S), which understates subset metrics.
    sub = sub.sample(frac=1, random_state=seed)
    n = 0
    with h5py.File(STEAD_HDF5, "r") as hf:
        for _, row in sub.iterrows():
            if max_n and n >= max_n:
                break
            tn = str(row["trace_name"])
            try:
                wf = np.array(hf["data"][tn]).astype(np.float32)
            except KeyError:
                continue
            if wf.shape != (6000, 3):
                continue
            if category == "earthquake_local":
                try:
                    p = int(round(float(row["p_arrival_sample"])))
                    s = int(round(float(row["s_arrival_sample"])))
                except (TypeError, ValueError):
                    continue
                if not (0 <= p < s < 6000):
                    continue
                yield wf, (p, s)   # sample indices
            else:
                yield wf, None
            n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--max-eq", type=int, default=2000)
    ap.add_argument("--max-noise", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=0.5, help="match tolerance [s]")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed for the subset")
    ap.add_argument("--pad", choices=["noise", "zero"], default="noise",
                    help="EQ window: 'noise' = spectrum-matched 90s window, P at "
                         "10%% (matches previous --mode static); 'zero' = "
                         "predict_array short-path (zero-pad to 90s)")
    args = ap.parse_args()

    pred = REDPANPredictor.from_checkpoint(
        args.ckpt, device="cuda" if torch.cuda.is_available() else "cpu")
    test_ids = set(np.load(STEAD_TEST_IDS, allow_pickle=True).tolist())

    def infer_noise(wf):
        # noise: 6000 -> predict_array short-path (zero-pad to 90s), one forward
        picker, detector = pred.predict_array(preprocess(wf), mode="single")
        return detector[:, 0], picker[:, 0], picker[:, 1]   # mask, P, S

    def infer_eq(wf, p_abs, s_abs):
        wf_p = preprocess(wf)
        if args.pad == "zero":                         # short-path zero-pad
            picker, detector = pred.predict_array(wf_p, mode="single")
            return detector[:, 0], picker[:, 0], picker[:, 1]
        # spectrum-matched 90s window (matches previous --mode static); the model
        # still runs ONE forward pass. Map the real region back to trace frame.
        win, rs, re, fp = build_static_window_eq(wf_p, p_abs, s_abs, in_samples=pred.pred_npts)
        picker, detector = pred.predict_array(win, mode="single")
        n, rn = wf_p.shape[0], re - rs
        mask = np.zeros(n, np.float32)
        P = np.zeros(n, np.float32)
        S = np.zeros(n, np.float32)
        mask[rs:re] = detector[fp:fp + rn, 0]
        P[rs:re] = picker[fp:fp + rn, 0]
        S[rs:re] = picker[fp:fp + rn, 1]
        return mask, P, S

    # ---- EARTHQUAKES: recall + residuals + precision bookkeeping ----
    p_res, s_res = [], []
    p_tp = s_tp = eq_det = n_eq = 0
    for wf, (p_abs, s_abs) in iter_stead("earthquake_local", args.max_eq, test_ids, args.seed):
        n_eq += 1
        lp, ls = p_abs * DT, s_abs * DT
        bp = best_pick(*infer_eq(wf, p_abs, s_abs))
        if bp is None:
            continue
        eq_det += 1
        pp, ss, _ = bp
        if abs(pp - lp) <= args.tol:
            p_tp += 1
            p_res.append(pp - lp)
        if abs(ss - ls) <= args.tol:
            s_tp += 1
            s_res.append(ss - ls)

    # ---- NOISE: false positives ----
    n_noise = noise_fp = 0
    for wf, _ in iter_stead("noise", args.max_noise, test_ids, args.seed):
        n_noise += 1
        if best_pick(*infer_noise(wf)) is not None:
            noise_fp += 1

    # ---- report ----
    # FP for a phase = mislocated EQ detections (det but pick wrong) + noise dets
    p_prec, p_rec, p_f1 = prf(p_tp, n_eq, (eq_det - p_tp) + noise_fp)
    s_prec, s_rec, s_f1 = prf(s_tp, n_eq, (eq_det - s_tp) + noise_fp)
    ps, ss_ = stats(p_res), stats(s_res)

    print(f"\nSTEAD single-forward-pass benchmark  (model={Path(args.ckpt).parent.name})")
    print(f"  EQ window: {args.pad}-pad{' (= previous --mode static)' if args.pad == 'noise' else ''}")
    print(f"  EQ traces: {n_eq}   noise traces: {n_noise}   tol: {args.tol}s")
    print(f"  detection rate (any trigger on EQ): {eq_det/max(n_eq,1):.3f}")
    print(f"  noise FALSE-POSITIVE rate:          {noise_fp/max(n_noise,1):.3f}  ({noise_fp}/{n_noise})")
    print("\n  phase |  precision  recall    F1   |   MAE     MAD     std    (n)")
    print("  ------+----------------------------+---------------------------------")
    print(f"   P    |   {p_prec:.3f}    {p_rec:.3f}  {p_f1:.3f}  |  {ps['MAE']:.3f}  {ps['MAD']:.3f}  {ps['std']:.3f}  ({ps['n']})")
    print(f"   S    |   {s_prec:.3f}    {s_rec:.3f}  {s_f1:.3f}  |  {ss_['MAE']:.3f}  {ss_['MAD']:.3f}  {ss_['std']:.3f}  ({ss_['n']})")
    print("  (residuals in seconds; MAE/MAD on within-tolerance picks)")


if __name__ == "__main__":
    main()
