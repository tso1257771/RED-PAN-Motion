"""Static (one window per record) inference and trigger extraction.

The logic is copied unchanged from the benchmark harness that produced the manuscript numbers
(``benchmark_unified_filt.py``); the input filter is an argument (``filt``) instead of the
``UNIFIED_FILTER`` environment variable, and whether the network takes ``z_raw`` is read from
its signature instead of retrying on ``TypeError``. Do not alter the logic: the published values
depend on it.
"""

from __future__ import annotations

import inspect

import numpy as np
import torch
from obspy.signal.trigger import trigger_onset
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

from redpan_motion.utils.waveform import (
    generate_matching_noise,
)  # identical to RED-PAN's redpan.utils

from .constants import DT, SENTINEL

_ACCEPTS_Z_RAW: dict[type, bool] = {}


def accepts_z_raw(net: torch.nn.Module) -> bool:
    """Whether ``net.forward`` takes ``z_raw`` (the 90 s models) or not (the 60 s RED-PAN).

    Read once per network class from the signature. The manuscript code called
    ``net(x, z_raw=...)`` and fell back to ``net(x)`` on ``TypeError``; this makes the same call
    for every model of the harness.
    """
    cls = type(net)
    if cls not in _ACCEPTS_Z_RAW:
        params = inspect.signature(net.forward).parameters.values()
        _ACCEPTS_Z_RAW[cls] = any(
            p.name == "z_raw" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params
        )
    return _ACCEPTS_Z_RAW[cls]


def preprocess_array(
    wf: np.ndarray,
    dt: float = DT,
    taper_ratio: float = 0.05,
    bandpass: tuple[float, float] = (1.0, 45.0),
    filt: str = "bp145",
) -> np.ndarray:
    """Detrend + Tukey taper + filter per channel. ``filt``: bp145 (1-45 Hz band-pass,
    RED-PAN and the SeisBench baselines) | hp1 (1 Hz high-pass, the 90 s models) | raw.
    Input/output shape ``(T, 3)``."""
    wf = wf.astype(np.float32).copy()
    for c in range(wf.shape[1]):
        wf[:, c] -= np.mean(wf[:, c])
    win = tukey(wf.shape[0], taper_ratio).astype(np.float32)
    wf *= win[:, None]
    fs = 1.0 / dt
    nyq = 0.5 * fs
    if filt == "hp1":
        sos = butter(4, 1.0 / nyq, btype="high", output="sos")
    elif filt == "raw":
        sos = None
    else:  # bp145 — original default
        sos = butter(4, [bandpass[0] / nyq, bandpass[1] / nyq], btype="band", output="sos")
    if sos is not None:
        for c in range(wf.shape[1]):
            wf[:, c] = sosfiltfilt(sos, wf[:, c]).astype(np.float32)
    return wf


def per_channel_norm(window: np.ndarray) -> np.ndarray:
    """``(x - mean)/std`` per channel — same as model.annotate_batch_pre."""
    out = window.astype(np.float32, copy=True)
    out -= out.mean(axis=-1, keepdims=True)
    std = out.std(axis=-1, keepdims=True)
    out = out / (std + 1e-10)
    return out


def restore_taper(
    wf_TC: np.ndarray, dt: float = DT, win_s: float = 0.5, threshold_ratio: float = 0.7
) -> np.ndarray:
    """Detect and replace pre-applied amplitude tapering at trace boundaries.

    Some external datasets (notably GeoNet noise) ship traces that have a
    Tukey-style envelope already applied. The slow amplitude rise at trace
    start mimics a P-wave onset and triggers the model under sliding-window
    inference. This helper:

      1. Computes RMS in ``win_s``-second bins on the Z channel.
      2. Finds the first/last bin whose RMS exceeds
         ``threshold_ratio * median(RMS)`` — the un-tapered span.
      3. Replaces the tapered samples (``[0:start]`` and ``[end:n]``) with
         spectrum-matched noise generated from the un-tapered middle, on
         each channel independently.

    Returns a new array (does not mutate input). If no taper is detected,
    returns a copy.
    """
    n, n_ch = wf_TC.shape
    win = max(1, int(round(win_s / dt)))
    z = wf_TC[:, 2]
    if n < win + 1:
        return wf_TC.copy()
    rms = np.array([np.sqrt(np.mean(z[i : i + win] ** 2)) for i in range(0, n - win, win)])
    if len(rms) == 0:
        return wf_TC.copy()
    median_rms = float(np.median(rms))
    above = rms > threshold_ratio * median_rms
    if not above.any():
        return wf_TC.copy()
    first_bin = int(np.argmax(above))
    last_bin = len(rms) - int(np.argmax(above[::-1]))
    start = first_bin * win
    end = min(last_bin * win, n)
    if start <= 0 and end >= n:
        return wf_TC.copy()
    out = wf_TC.copy()
    mid = wf_TC[start:end]
    if mid.shape[0] < 10:
        # un-tapered region too small to estimate spectrum — bail out
        return out
    if start > 0:
        for ch in range(n_ch):
            out[:start, ch] = generate_matching_noise(mid[:, ch], start).astype(np.float32)
    if end < n:
        for ch in range(n_ch):
            out[end:, ch] = generate_matching_noise(mid[:, ch], n - end).astype(np.float32)
    return out


def _build_static_window_eq(
    wf_TC: np.ndarray,
    p_abs: int,
    s_abs: int,
    in_samples: int,
    p_pos_frac: float = 0.1,
    coda_offset_s: float = 2.0,
) -> tuple[np.ndarray, int, int, int]:
    """Place P at p_pos_frac of the in_samples window (default 10% = 900 samples
    @ 9000-sample context). Front-pad with spectrum-matched noise from the
    pre-P portion ``wf[:p_abs]``; back-pad with spectrum-matched noise from
    the post-S coda ``wf[s_abs + coda_offset_s/dt:]``.

    Returns ``(window, real_start, real_end, front_pad_n)``:
      - window: shape ``(in_samples, 3)`` float32
      - real_start, real_end: indices into the original ``wf_TC`` covered by
        the real (un-synthesised) portion of the window
      - front_pad_n: number of synthetic samples prepended to the front
        (used to map model output back to the original trace coordinates).
    """
    n, n_ch = wf_TC.shape
    target_p_local = int(round(p_pos_frac * in_samples))
    desired_start = p_abs - target_p_local
    desired_end = desired_start + in_samples

    front_pad_n = max(0, -desired_start)
    back_pad_n = max(0, desired_end - n)
    real_start = max(0, desired_start)
    real_end = min(n, desired_end)
    real_window = wf_TC[real_start:real_end]

    # Front pad: spectrum-match against pre-P noise wf[:p_abs]. If p_abs is
    # too small (P close to start of trace), fall back to whatever clean
    # samples exist before the pick; if absolutely nothing, fall back to
    # zero.
    front_pad = np.zeros((front_pad_n, n_ch), dtype=np.float32)
    if front_pad_n > 0:
        ref_pre = wf_TC[: max(0, p_abs)]
        if len(ref_pre) >= 10:
            for ch in range(n_ch):
                front_pad[:, ch] = generate_matching_noise(ref_pre[:, ch], front_pad_n).astype(
                    np.float32
                )
        # else: leave as zeros (degenerate case — P too close to trace start)

    # Back pad: spectrum-match against post-S coda wf[s_abs + coda:].
    coda_offset_n = int(round(coda_offset_s / DT))
    back_pad = np.zeros((back_pad_n, n_ch), dtype=np.float32)
    if back_pad_n > 0:
        ref_start = max(0, min(n - 100, s_abs + coda_offset_n))
        ref_post = wf_TC[ref_start:]
        if len(ref_post) >= 10:
            for ch in range(n_ch):
                back_pad[:, ch] = generate_matching_noise(ref_post[:, ch], back_pad_n).astype(
                    np.float32
                )

    win = np.concatenate([front_pad, real_window, back_pad], axis=0).astype(np.float32)
    assert win.shape[0] == in_samples, (win.shape, front_pad_n, back_pad_n, real_end - real_start)
    return win, real_start, real_end, front_pad_n


def run_static(
    model,
    wf_TC: np.ndarray,
    in_samples: int,
    device: str = "cpu",
    p_abs: int | None = None,
    s_abs: int | None = None,
    p_pos_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """One-shot inference. wf_TC shape (T, 3).

    EQ samples (``p_abs``, ``s_abs`` both >= 0): place P at ``p_pos_frac`` of
    the model window (default 10% = 900 samples @ 9000); front-pad with
    spectrum-matched noise from ``wf[:p_abs]``, back-pad with spectrum-matched
    noise from ``wf[s_abs + 2 s:]``.

    Noise samples (``p_abs``/``s_abs`` is None or < 0): fall back to the
    legacy zero-front-pad / centre-crop behaviour.

    Returns (mask, P, S, polarity?), each of length ``T`` (the original trace
    length). Predictions outside the real region of the constructed window are
    zero (we never reuse synthetic-pad model output as if it were real)."""
    n = wf_TC.shape[0]
    is_eq = (p_abs is not None) and (s_abs is not None) and (p_abs >= 0) and (s_abs >= 0)

    if is_eq:
        win, real_start, real_end, front_pad_n = _build_static_window_eq(
            wf_TC,
            int(p_abs),
            int(s_abs),
            in_samples,
            p_pos_frac=p_pos_frac,
        )
    else:
        # Noise / unlabeled fallback (legacy behaviour)
        if n == in_samples:
            win = wf_TC
            real_start = 0
            real_end = n
            front_pad_n = 0
        elif n < in_samples:
            pad = np.zeros((in_samples - n, 3), dtype=np.float32)
            win = np.concatenate([wf_TC, pad], axis=0)
            real_start = 0
            real_end = n
            front_pad_n = 0
        else:
            real_start = (n - in_samples) // 2
            real_end = real_start + in_samples
            win = wf_TC[real_start:real_end]
            front_pad_n = 0

    win_cf = win.T  # (3, in_samples)
    win_cf = per_channel_norm(win_cf)
    x = torch.from_numpy(win_cf[None]).to(device)
    # z_raw: sign-preserved Z, max-abs scaled — must match training input to
    # the polarity stream. Without it, polarity outputs are wrong (see
    # continuous_3models.py for full rationale).
    z_seg = win.T[2]  # raw Z, no normalization
    z_max = float(np.max(np.abs(z_seg)))
    z_raw_np = z_seg / z_max if z_max > 1e-9 else z_seg
    z_raw_t = torch.from_numpy(z_raw_np[None, None].astype(np.float32)).to(device)
    with torch.no_grad():
        if accepts_z_raw(model.backbone.inner):
            out = model.backbone.inner(x, z_raw=z_raw_t)
        else:
            out = model.backbone.inner(x)
    picker = out[0]  # (1, 3, T) softmax(P, S, N)
    detector = out[-1]  # (1, 2, T) softmax or (1, 1, T) sigmoid
    polarity_tensor = out[1] if len(out) >= 3 else None
    mask = detector[0, 0].cpu().numpy()
    p_arr = picker[0, 0].cpu().numpy()
    s_arr = picker[0, 1].cpu().numpy()
    polarity = None
    if polarity_tensor is not None:
        pol_np = polarity_tensor[0].cpu().numpy()
        # Channel 0 for every head shape (as in the manuscript code). For the 3-class [N, U, D]
        # softmax of edge / rpm this is P(None), so the ``polarity_at_P`` column holds P(None) at
        # the P pick, not a signed polarity. No scorer reads that column (Table V uses
        # run_polarity.py); the name is kept so the CSVs match the archived outputs.
        polarity = pol_np[0]

    # Map predictions on the (possibly padded) window back to original trace
    # coordinates of length n. Real region of the window spans
    # win[front_pad_n : front_pad_n + (real_end - real_start)].
    full_mask = np.zeros(n, dtype=np.float32)
    full_p = np.zeros(n, dtype=np.float32)
    full_s = np.zeros(n, dtype=np.float32)
    full_pol: np.ndarray | None = np.zeros(n, dtype=np.float32) if polarity is not None else None
    real_n = real_end - real_start
    full_mask[real_start:real_end] = mask[front_pad_n : front_pad_n + real_n]
    full_p[real_start:real_end] = p_arr[front_pad_n : front_pad_n + real_n]
    full_s[real_start:real_end] = s_arr[front_pad_n : front_pad_n + real_n]
    if full_pol is not None and polarity is not None:
        full_pol[real_start:real_end] = polarity[front_pad_n : front_pad_n + real_n]
    return full_mask, full_p, full_s, full_pol


def smooth(x: np.ndarray, npts: int) -> np.ndarray:
    """Moving average over ``npts`` samples (``np.convolve`` mode ``same``)."""
    if npts <= 1:
        return x
    kernel = np.ones(npts, dtype=np.float32) / npts
    return np.convolve(x, kernel, mode="same")


def extract_triggers(
    mask: np.ndarray,
    p_arr: np.ndarray,
    s_arr: np.ndarray,
    polarity: np.ndarray | None,
    dt: float = DT,
    thr_on: float = 0.1,
    thr_off: float = 0.1,
    smooth_npts: int = 10,
) -> list[dict]:
    """For every (mask) trigger window, compute trigger-relative pick + mask
    metrics. Returns one dict per trigger; empty list if no triggers.

    Note: obspy's ``trigger_onset`` returns the last active sample as ``hi`` (inclusive), and the
    slices below are ``[lo:hi]``, so that sample is left out of the mask statistics and the P / S
    argmax (and single-sample triggers, ``hi == lo``, are skipped). The effect is at most one
    sample; it is kept so that the published values reproduce (README, Known caveats)."""
    sm = smooth(mask, smooth_npts)
    triggers = trigger_onset(sm, thr_on, thr_off)
    out = []
    if len(triggers) == 0:
        return out

    for k, (lo, hi) in enumerate(triggers):
        lo, hi = int(lo), int(hi)
        if hi <= lo:
            continue
        m_seg = mask[lo:hi]
        p_seg = p_arr[lo:hi]
        s_seg = s_arr[lo:hi]
        p_idx_local = int(np.argmax(p_seg))
        s_idx_local = int(np.argmax(s_seg))
        p_idx = lo + p_idx_local
        s_idx = lo + s_idx_local
        rec = dict(
            trigger_idx=k,
            trigger_on_sec=lo * dt,
            trigger_off_sec=hi * dt,
            mask_peak=float(np.max(m_seg)),
            mask_mean=float(np.mean(m_seg)),
            P_pick_sec=p_idx * dt,
            P_pick_prob=float(p_seg[p_idx_local]),
            S_pick_sec=s_idx * dt,
            S_pick_prob=float(s_seg[s_idx_local]),
            polarity_at_P=float(polarity[p_idx]) if polarity is not None else float("nan"),
        )
        out.append(rec)
    return out


def build_rows(
    rec: dict,
    triggers: list[dict],
    labelP_sec: float,
    labelS_sec: float,
    polarity_label: str | None,
) -> list[dict]:
    """Convert one (sample, [triggers]) pair into one or more CSV rows."""
    common = dict(
        evid=rec["evid"],
        label_type=rec["label_type"],
        mode=rec["mode"],
        model=rec["model"],
        labelP_sec=labelP_sec,
        labelS_sec=labelS_sec,
        ps_diff_sec=(labelS_sec - labelP_sec)
        if (labelP_sec >= 0 and labelS_sec >= 0)
        else SENTINEL,
        polarity_label=polarity_label or "",
        n_triggers=len(triggers),
    )
    if not triggers:
        return [
            {
                **common,
                "trigger_idx": -1,
                "trigger_on_sec": SENTINEL,
                "trigger_off_sec": SENTINEL,
                "mask_peak": 0.0,
                "mask_mean": 0.0,
                "P_pick_sec": SENTINEL,
                "P_pick_prob": 0.0,
                "S_pick_sec": SENTINEL,
                "S_pick_prob": 0.0,
                "polarity_at_P": float("nan"),
                "P_residual_sec": float("nan"),
                "S_residual_sec": float("nan"),
            }
        ]
    rows = []
    for t in triggers:
        p_res = (t["P_pick_sec"] - labelP_sec) if labelP_sec >= 0 else float("nan")
        s_res = (t["S_pick_sec"] - labelS_sec) if labelS_sec >= 0 else float("nan")
        rows.append(
            {
                **common,
                **t,
                "P_residual_sec": p_res,
                "S_residual_sec": s_res,
            }
        )
    return rows
