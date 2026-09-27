"""Mask-gated P-S pairing, with no dependency on amplitudes or response.

``picks.py`` re-exports ``detect_events_joint``, so
``from redpan_motion.picks import detect_events_joint`` keeps working. It lives
here so that the SeisBench integration can pair phases without importing the
Wood-Anderson amplitude and instrument-response layers, which it never uses.
"""
from __future__ import annotations

import numpy as np
from obspy.signal.trigger import trigger_onset
from scipy.signal import find_peaks

from redpan_motion.sp_thresholds import (
    _SP_SMOOTH_NPTS as sp_smooth_npts,
    _SP_TRIG_ONSET as sp_trig_onset,
    floor_thresholds as sp_floor_thresholds,
    params_for_duration as sp_params_for_duration,
)


def detect_events_joint(
    p_prob: np.ndarray,
    s_prob: np.ndarray,
    mask_prob: np.ndarray,
    dt: float,
    p_pick_thr: float,
    s_pick_thr: float,
    det_thr: float,
    min_mask_sec: float = 0.5,
    pad_sec: float = 1.0,
    min_peak_dist_sec: float = 1.0,
    sp_thresholds: dict | None = None,
) -> list[tuple[int, float, int, float]]:
    """P-S pairing driven by mask detection — matches RED-PAN ``picker_info``.

    The mask DETECTION decides the pairs (not the P/S peaks on their own):
      1. find_peaks on P (height >= p_pick_thr) and S (height >= s_pick_thr),
         each with >= ``min_peak_dist_sec`` spacing.
      2. ``trigger_onset(mask, det_thr)`` -> detections; drop any shorter than
         ``min_mask_sec``.
      3. For each detection, consider only the P/S peaks falling within the
         detection span padded by ``pad_sec`` on each side, and take the strongest
         P and strongest S. A detection with BOTH a P and an S peak yields exactly
         one (P, S) pair; detections missing either are dropped.

    ``sp_thresholds`` (a table resolved via ``redpan_motion.sp_thresholds``) turns on
    S-P-adaptive gating: each detection's mask-trigger DURATION (an S-P proxy) selects a
    ``(mask_mean, P, S)`` triple. In this mode the trigger is re-delineated with the table's
    CALIBRATED recipe (smoothed mask, ``_SP_TRIG_ONSET`` onset — identical to the benchmark
    ``extract_triggers``, so duration and mask-mean match the fit), ``p_pick_thr`` /
    ``s_pick_thr`` drop to the table's FLOOR (candidate peaks are kept), then per detection
    the mask MEAN over the trigger must clear the bin's mask threshold and the P/S peaks must
    clear the bin's P/S thresholds. The gate is the mask MEAN (not peak): a wide long-S-P
    trigger demands sustained detection, rejecting brief in-window noise spikes. ``det_thr``
    delineates triggers only in the non-adaptive path. ``None`` (default) = fixed-threshold
    behaviour, unchanged.

    Returns ``[(p_idx, p_peak, s_idx, s_peak), ...]`` ordered by detection."""
    pad = int(pad_sec / dt)
    min_len = int(min_mask_sec / dt)
    min_dist = max(1, int(min_peak_dist_sec / dt))
    if sp_thresholds is not None:
        _, p_floor, s_floor = sp_floor_thresholds(sp_thresholds)   # keep candidates at the floor
        p_pick_thr = min(p_pick_thr, p_floor)
        s_pick_thr = min(s_pick_thr, s_floor)
        # Delineate triggers exactly like the benchmark extract_triggers the thresholds were
        # fit on: smoothed mask (float32 mean kernel) + low onset. The mean gate is
        # window-sensitive, so it only transfers under this recipe.
        if sp_smooth_npts > 1:
            kernel = np.ones(sp_smooth_npts, dtype=np.float32) / sp_smooth_npts
            m_trig = np.convolve(mask_prob, kernel, mode="same")
        else:
            m_trig = mask_prob
        trig_on = trig_off = sp_trig_onset
    else:
        m_trig, trig_on, trig_off = mask_prob, det_thr, det_thr
    p_peaks = find_peaks(p_prob, height=p_pick_thr, distance=min_dist)[0]
    s_peaks = find_peaks(s_prob, height=s_pick_thr, distance=min_dist)[0]
    pairs = []
    for on, off in trigger_onset(m_trig, trig_on, trig_off):
        on, off = int(on), int(off)
        if off - on < min_len:                       # drop short mask detections
            continue
        p_thr, s_thr = p_pick_thr, s_pick_thr
        if sp_thresholds is not None:                # S-P-adaptive: pick thresholds by duration
            m_thr, p_thr, s_thr = sp_params_for_duration((off - on) * dt, sp_thresholds)
            if mask_prob[on:off].mean() < m_thr:     # mask MEAN over the trigger [on:off]
                continue
        lo, hi = on - pad, off + pad                  # detection span padded +/- pad_sec
        pp = p_peaks[(p_peaks >= lo) & (p_peaks <= hi)]
        pp = pp[p_prob[pp] >= p_thr]
        sp = s_peaks[(s_peaks >= lo) & (s_peaks <= hi)]
        sp = sp[s_prob[sp] >= s_thr]
        if len(pp) == 0 or len(sp) == 0:              # need BOTH a P and an S in the detection
            continue
        p_idx = int(pp[np.argmax(p_prob[pp])])
        s_idx = int(sp[np.argmax(s_prob[sp])])
        pairs.append((p_idx, float(p_prob[p_idx]), s_idx, float(s_prob[s_idx])))
    return pairs
