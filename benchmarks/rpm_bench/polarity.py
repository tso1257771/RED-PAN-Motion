"""CEED first-motion polarity benchmark helpers (Table V). The logic is copied unchanged from the
harness that produced the manuscript numbers (``benchmark_ceed_polarity_v13_xl.py``).

Per record: one window of 9,000 samples centred on the catalog P (zero-filled past the record),
no filter, per-channel z-score for the picker/detector input, raw vertical scaled by its maximum
(sign kept) for the polarity input; polarity read at the labeled P sample.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from obspy.signal.trigger import trigger_onset


def crop_to_window(wf, p_sample, target_len):
    """Cut ``target_len`` samples of ``wf`` (3, T) with ``p_sample`` at ``target_len // 2``,
    zero-filled where the window leaves the record. Returns (window, P index in it, start)."""
    n = wf.shape[1]
    half = target_len // 2
    start = p_sample - half
    end = start + target_len
    out = np.zeros((3, target_len), dtype=wf.dtype)
    src_lo = max(0, start)
    src_hi = min(n, end)
    dst_lo = src_lo - start
    dst_hi = dst_lo + (src_hi - src_lo)
    out[:, dst_lo:dst_hi] = wf[:, src_lo:src_hi]
    return out, p_sample - start, start


def open_year_files(cache_dir, regions_years):
    """Open ``waveforms<region><year>.hdf5`` for each pair that exists; {(region, year): File}."""
    handles = {}
    for region, year in regions_years:
        path = Path(cache_dir) / f"waveforms{region}{year}.hdf5"
        if path.exists():
            handles[(region, year)] = h5py.File(path, "r")
    return handles


def build_polarity_index(cache_dir):
    """{trace_name: U | D | N} from the ``trace_p_polarity`` column of the metadata CSVs."""
    idx = {}
    cache_dir = Path(cache_dir)
    csvs = sorted(cache_dir.glob("metadata*.csv"))
    logging.info("Scanning %d metadata CSVs in %s", len(csvs), cache_dir)
    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path, usecols=["trace_name", "trace_p_polarity"], low_memory=False)
        except (KeyError, ValueError):
            continue
        df = df.dropna(subset=["trace_p_polarity"])
        df = df[df["trace_p_polarity"].isin(["U", "D", "N"])]
        for tn, pol in zip(df["trace_name"].values, df["trace_p_polarity"].values):
            idx[tn] = pol
    logging.info("Built polarity index with %d labeled traces", len(idx))
    counts = pd.Series(list(idx.values())).value_counts()
    logging.info("  distribution: %s", dict(counts))
    return idx


def decode_polarity(pol_tensor, p_idx, imp_score, imp_thr, mode):
    """(class, signed score, P(U), P(D)) of the polarity output at sample ``p_idx``."""
    if mode == "softmax_ce":
        # 3-ch [N, U, D] softmax — abstention is the N class itself (argmax over the 3).
        n = float(pol_tensor[0, p_idx].item())
        u = float(pol_tensor[1, p_idx].item())
        d = float(pol_tensor[2, p_idx].item())
        signed = u - d
        if n >= u and n >= d:
            pred = "N"
        else:
            pred = "U" if u >= d else "D"
        return pred, signed, u, d
    if mode == "bce":
        s = float(pol_tensor[0, p_idx].item())
        u = (s + 1.0) / 2.0
        d = 1.0 - u
        signed = s
    else:  # softmax_ud — 2-ch [U, D]; abstention gated by the separate impulsive head
        u = float(pol_tensor[0, p_idx].item())
        d = float(pol_tensor[1, p_idx].item())
        signed = u - d
    if not np.isnan(imp_score) and imp_score > imp_thr:
        pred = "U" if u >= d else "D"
    else:
        pred = "N"
    return pred, signed, u, d


def _smooth(x, n=10):
    if n <= 1:
        return x
    pad = n // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(n - pad - 1, x[-1])])
    kernel = np.ones(n) / n
    return np.convolve(xp, kernel, mode="valid")


def extract_best_pick(arr_tensor_1d_T, lo, hi):
    """Return (idx_in_window, prob) for argmax of arr inside [lo,hi)."""
    if hi <= lo:
        return -1, 0.0
    seg = arr_tensor_1d_T[lo:hi]
    if seg.numel() == 0:
        return -1, 0.0
    rel = int(seg.argmax().item())
    return lo + rel, float(seg[rel].item())


def detection_picking(
    picker, detector, p_label, s_label, in_samples, thr_on=0.3, thr_off=0.3, smooth_npts=10
):
    """Run trigger_onset on the smoothed detector mask; for each trigger that
    overlaps the labeled (P,S) window, record peak/mean and the argmax P/S
    inside that trigger. Returns dict.

    Falls back to "no trigger" when the model fires nothing above the
    threshold."""
    mask = detector[0, 0].cpu().numpy()
    smask = _smooth(mask, smooth_npts)
    triggers = trigger_onset(smask, thr_on, thr_off)
    out = dict(
        n_triggers=len(triggers),
        own_P_idx=-1,
        own_P_prob=0.0,
        own_S_idx=-1,
        own_S_prob=0.0,
        mask_peak=0.0,
        mask_mean=0.0,
        trigger_on=-1,
        trigger_off=-1,
    )
    if len(triggers) == 0:
        return out
    # Pick the trigger whose window most overlaps [p_label - 50, s_label + 100]
    # samples; falls back to the highest-mean trigger.
    label_lo = max(0, p_label - 50)
    label_hi = min(in_samples, s_label + 100)
    # Note: trigger_onset returns the last active sample as hi (inclusive) and the slices are
    # mask[lo:hi], so that sample is left out (at most one sample; kept for reproduction).
    best_score = -1.0
    best = None
    for lo, hi in triggers:
        lo = int(lo)
        hi = int(hi)
        overlap = max(0, min(hi, label_hi) - max(lo, label_lo))
        m_seg = mask[lo:hi]
        if len(m_seg) == 0:
            continue
        score = overlap + 0.001 * float(m_seg.mean())
        if score > best_score:
            best_score = score
            best = (lo, hi)
    if best is None:
        # Reached only when every trigger is a single sample (trigger_onset gives hi == lo, so
        # mask[lo:hi] is empty): the first trigger is taken and its statistics are zero.
        # Fallback: highest mean
        best = max(
            ((int(lo), int(hi)) for lo, hi in triggers),
            key=lambda lh: float(mask[lh[0] : lh[1]].mean()) if lh[1] > lh[0] else -1,
        )
    lo, hi = best
    m_seg = mask[lo:hi]
    out.update(
        mask_peak=float(m_seg.max() if len(m_seg) else 0.0),
        mask_mean=float(m_seg.mean() if len(m_seg) else 0.0),
        trigger_on=lo,
        trigger_off=hi,
    )
    p_idx, p_prob = extract_best_pick(picker[0, 0], lo, hi)
    s_idx, s_prob = extract_best_pick(picker[0, 1], lo, hi)
    out.update(own_P_idx=p_idx, own_P_prob=p_prob, own_S_idx=s_idx, own_S_prob=s_prob)
    return out
