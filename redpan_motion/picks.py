"""Pick detection, mask-detection P-S pairing, and the redpan_picks DataFrame.

Operates on the model's output arrays; amplitudes come from
``redpan_motion.amplitudes`` / ``redpan_motion.response``.

    detect_events_joint(p, s, mask, dt, ...)  -> [(p_idx, p_peak, s_idx, s_peak)]
    picks_to_dataframe(picker, detector, polarity, t0, station_id, ...) -> DataFrame
"""
import numpy as np
import pandas as pd

from redpan_motion.amplitudes import (
    adaptive_s_window,
    compute_snr,
    full_response_amplitudes,
    window_amplitudes,
)

# Moved to redpan_motion.pairing so the pairing carries no amplitude dependency.
# Re-exported here because callers and __init__ use this name.
from redpan_motion.pairing import detect_events_joint
from redpan_motion.response import (
    MAG_WINDOW_ALPHA,
    MAG_WINDOW_PRE_P,
    MAG_WINDOW_TMAX,
    MAG_WINDOW_TMIN_RATIO,
    WADATI_VPVS_FACTOR,
    sensor_type as _sensor_type,
)
from redpan_motion.sp_thresholds import resolve_table as sp_resolve_table

# The 15-column redpan_picks schema.
COLUMNS = [
    "id", "timestamp", "prob", "type", "pick_idx",
    "amp_disp_nm_sensonly", "amp_vel_nm_s_sensonly", "amp_acc_gal_sensonly",
    "amp_disp_nm", "amp_vel_nm_s", "amp_acc_gal",
    "WA_amp_disp_mm", "snr",
    "polarity", "polarity_prob",   # first motion at P: +1 up / -1 down, softmax prob
]


def s_match_window(sp_diff_sec, base_sec=0.5, frac=0.10):
    """Adaptive S-arrival matching tolerance (s): ``max(base_sec, frac * S-P time)``.

    The S onset smears with epicentral distance (which scales with the S-P time), so a
    real detected S can land seconds off the theoretical/catalogue S for distant events.
    This widens the matching window proportionally while staying tight (``base_sec``) for
    local, short-S-P events. Vectorized: ``sp_diff_sec`` may be a scalar or array.

    Use in the ASSOCIATION stage (matching a detected S to a theoretical S from a velocity
    model or P + S-P time), NOT for detection — the S residual is an association criterion,
    so loosening it costs ZERO false positives (noise FP is probability-based).
    """
    return np.maximum(base_sec, frac * np.asarray(sp_diff_sec, dtype=float))


def match_s_arrival(detected_s_sec, theoretical_s_sec, sp_diff_sec, base_sec=0.5, frac=0.10):
    """True where a detected S matches a theoretical S within ``s_match_window`` (vectorized).
    ``sp_diff_sec`` is the S-P time (theoretical S minus P) driving the window width."""
    return (np.abs(np.asarray(detected_s_sec, float) - np.asarray(theoretical_s_sec, float))
            <= s_match_window(sp_diff_sec, base_sec, frac))




def picks_to_dataframe(picker, detector, polarity, starttime, station_id,
                       raw_counts=None, wf_sensonly=None, inv=None, dt=0.01,
                       amplitude=True,
                       p_pick_thr=0.30, s_pick_thr=0.20, det_thr=0.50,
                       min_sp_sec=1.0, sp_thresholds=None,
                       post_s_factor=1.5, max_amp_win_sec=80.0, noise_win_sec=3.0):
    """Convert raw model output into the 15-column redpan_picks DataFrame.

    Receives the ``REDPANPredictor.predict_arrays`` output directly:
      picker   (T, 3) softmax [P, S, Noise]
      detector (T, 2) softmax [mask, unmask]
      polarity (T, 3) softmax [N, U, D]  (or 2-/1-channel head, or None)
    Post-processing is mask-detection P-S pairing (``detect_events_joint``,
    matching RED-PAN ``picker_info``): each mask detection (>= ``det_thr``) that
    contains both a P peak (>= ``p_pick_thr``) and an S peak (>= ``s_pick_thr``)
    yields one (P, S) pair. With ``amplitude=False`` the per-pick response removal
    is skipped (amp/WA/snr -> NaN; raw_counts/wf_sensonly/inv unused)."""
    p_prob, s_prob = picker[:, 0], picker[:, 1]   # [P, S, Noise]
    mask_prob = detector[:, 0]                     # [mask, unmask]
    polarity_arr = polarity
    npts = len(mask_prob)
    parts = station_id.split(".")
    if len(parts) != 4:
        raise ValueError(f"station_id must be 'NET.STA.LOC.CHN2', got {station_id!r}")
    net, sta, loc, chn_pre = parts
    sensor = _sensor_type(station_id)
    noise_npts = max(1, int(noise_win_sec / dt))
    pairs = detect_events_joint(p_prob, s_prob, mask_prob, dt,
                                p_pick_thr, s_pick_thr, det_thr,
                                sp_thresholds=sp_resolve_table(sp_thresholds))
    # Require S at least min_sp_sec after P (== RED-PAN _matches_to_dataframe's
    # `s_idx - p_idx < 100` skip): drops degenerate / out-of-order pairs whose
    # ~1-sample amplitude window crashes np.gradient (velocity sensors), and
    # matches the reference pairing. Floor of 2 samples guarantees a safe window.
    min_sp = max(2, round(min_sp_sec / dt))
    pairs = [pair for pair in pairs if pair[2] - pair[0] >= min_sp]
    nan = float("nan")
    rows = []
    for pair_idx, (p_idx, p_pv, s_idx, s_pv) in enumerate(pairs):
        # first-motion polarity at the P pick (always; cheap). Channel layout:
        #   3-ch softmax [N,U,D] (shipped), 2-ch softmax [U,D], or 1-ch signed tanh.
        if (polarity_arr is not None and polarity_arr.ndim == 2
                and 0 <= p_idx < len(polarity_arr)):
            pol_ch = polarity_arr.shape[1]
            if pol_ch >= 3:        # [N, U, D]
                up_prob, dn_prob = float(polarity_arr[p_idx, 1]), float(polarity_arr[p_idx, 2])
                pol_sign, pol_prob = (1 if up_prob >= dn_prob else -1), round(max(up_prob, dn_prob), 3)
            elif pol_ch == 2:      # [U, D]
                up_prob, dn_prob = float(polarity_arr[p_idx, 0]), float(polarity_arr[p_idx, 1])
                pol_sign, pol_prob = (1 if up_prob >= dn_prob else -1), round(max(up_prob, dn_prob), 3)
            else:                  # signed tanh, single channel in [-1, 1]
                val = float(polarity_arr[p_idx, 0])
                pol_sign, pol_prob = (1 if val >= 0 else -1), round(abs(val), 3)
        else:
            pol_sign = pol_prob = nan

        if amplitude:
            sp_samples = s_idx - p_idx
            p_win = max(1, sp_samples)
            s_end = min(s_idx + int(post_s_factor * sp_samples), p_idx + int(max_amp_win_sec / dt), npts)
            s_win_nom = max(1, s_end - s_idx)
            # amplitude-decay-based S window (shorten to drop coda/noise spikes)
            s_win = adaptive_s_window(wf_sensonly, s_idx, s_win_nom, p_idx, dt)
            # Wadati-proxy magnitude window for Wood-Anderson (reamp convention)
            sp_sec = sp_samples * dt
            hyp_km = WADATI_VPVS_FACTOR * sp_sec
            t_post_s = min(MAG_WINDOW_TMIN_RATIO * sp_sec + MAG_WINDOW_ALPHA * hyp_km,
                           MAG_WINDOW_TMAX)
            mag_start = max(0, p_idx - int(MAG_WINDOW_PRE_P / dt))
            mag_end = min(npts, s_idx + int(t_post_s / dt))
            if mag_end <= mag_start:
                mag_end = min(npts, mag_start + max(1, p_win + s_win))
            # full ObsPy response (both phases + Wood-Anderson) in a single call
            wa_mm, p_full, s_full = full_response_amplitudes(
                raw_counts, p_idx, p_win, s_idx, s_win,
                mag_start=mag_start, mag_end=mag_end, inv=inv,
                net=net, sta=sta, loc=loc, chn_pre=chn_pre, t0=starttime,
            )
            phase_rows = (("p", p_idx, p_pv, p_win, p_full),
                          ("s", s_idx, s_pv, s_win, s_full))
        else:                               # CWA-style fast path: no amplitudes
            wa_mm = nan
            phase_rows = (("p", p_idx, p_pv, None, None), ("s", s_idx, s_pv, None, None))

        for phase, idx, pv, win, full in phase_rows:
            if amplitude:
                disp_so, vel_so, acc_so = window_amplitudes(wf_sensonly, idx, win, sensor, dt)
                disp_fr, vel_fr, acc_fr = full
                anchor = p_idx if phase == "s" else None
                snr = compute_snr(wf_sensonly, idx, win, noise_npts, noise_anchor=anchor)
                snr_v = round(snr, 2) if not np.isnan(snr) else snr
            else:
                disp_so = vel_so = acc_so = disp_fr = vel_fr = acc_fr = snr_v = nan
            rows.append({
                "id": station_id,
                "timestamp": str(starttime + idx * dt)[:24],
                "prob": round(pv, 2),
                "type": phase,
                "pick_idx": pair_idx,
                "amp_disp_nm_sensonly": disp_so, "amp_vel_nm_s_sensonly": vel_so,
                "amp_acc_gal_sensonly": acc_so,
                "amp_disp_nm": disp_fr, "amp_vel_nm_s": vel_fr, "amp_acc_gal": acc_fr,
                "WA_amp_disp_mm": wa_mm,
                "snr": snr_v,
                "polarity": pol_sign if phase == "p" else nan,
                "polarity_prob": pol_prob if phase == "p" else nan,
            })
    return pd.DataFrame(rows, columns=COLUMNS)
