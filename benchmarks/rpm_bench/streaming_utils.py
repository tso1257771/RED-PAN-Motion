"""Streaming-replay helpers (the early-warning replay of the Discussion).

Two code paths, each with the logic copied unchanged from the script that produced the
manuscript values:

* 90 s models (Edge-RED-PAN-Motion, RED-PAN-Motion): ``eew_dynamic_pred_v49.py`` — windows padded
  with the pre-P median, per-channel demean + z-score, raw-Z polarity input, no filter.
* 60 s RED-PAN: ``P01_tf60_*.py`` / ``P02_tf60_*.py`` of the original RED-PAN dynamic-prediction
  benchmark — ``sac_len_complement`` and ``stream_standardize`` from ``REDPAN_tools.data_utils``
  (vendored below, logic verbatim), no filter.

The two paths differ in padding and normalization because they reproduce the two original
benchmarks; do not merge them. The 60 s noise path writes the P probability into the
``Mask_probability`` column (as the original P02 script did); the published 60 s replay values
include this, and ``run_streaming.py --noise-mask mask`` writes the mask instead.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from glob import glob

import numpy as np
import torch
from obspy.signal.trigger import trigger_onset
from scipy.signal import find_peaks

from .constants import DT  # 0.01 s, 100 Hz

log = logging.getLogger(__name__)

PRED_INTERVAL_SEC = 0.05  # streaming cadence: 20 inferences/sec
P_DETECT_THRE = 0.1  # find_peaks height
P_TRUE_PICK_THRE = 0.5  # |trg_p_utc - gt_p_utc| tolerance (s)
DELAY_THRE = 5  # discard picks more than this far behind window_end
LOOP_PRE_P_SEC = 1  # wf_ent starts P - 1
LOOP_POST_P_SEC = 5  # wf_ent ends P + 5
TRIGGER_ON_OFF = (0.05, 0.05)  # trigger_onset thresholds
TRIGGER_MAX_LEN = 150  # samples


# ──────────────────────────────────────────────────────────────────────────
# event lists (shared)
# ──────────────────────────────────────────────────────────────────────────
def list_eq_events(data_dir):
    """data_dir layout: data_dir/<EVID>/<station>.<comp>.sac, EVID = TSMIP_* or Palert_*."""
    return np.sort(
        np.hstack(
            [
                glob(os.path.join(data_dir, "TSMIP_????????.???")),
                glob(os.path.join(data_dir, "Palert_????????.???")),
            ]
        )
    )


def list_noise_subdirs(data_dir):
    """noise layout: data_dir/sac_120s_*/<EVID>/ (5-char EVID)."""
    return np.sort(
        np.hstack(
            [
                glob(os.path.join(data_dir, "sac_120s_CWB_StrongMotion", "?????")),
                glob(os.path.join(data_dir, "sac_120s_Palert", "?????")),
            ]
        )
    )


# ──────────────────────────────────────────────────────────────────────────
# 90 s path (eew_dynamic_pred_v49.py, run with --highpass 0)
# ──────────────────────────────────────────────────────────────────────────
def compute_pre_p_medians(stream, gt_p_utc, fs=100.0, min_samples=100):
    """Per-channel median of the PRE-P portion of the trace — the actual noise
    floor before the earthquake. Used as the pad value when sliding-windows
    extend before trace_start.

    Falls back to whole-trace median per channel if pre-P is too short (<1 s)
    or no gt_p_utc (noise mode)."""
    meds = np.zeros(3, dtype=np.float32)
    for ci in range(min(3, len(stream))):
        tr = stream[ci]
        if gt_p_utc is not None:
            pre_p_end_pt = int(round((gt_p_utc - tr.stats.starttime) * fs))
            if pre_p_end_pt >= min_samples:
                meds[ci] = float(np.median(tr.data[:pre_p_end_pt]))
                continue
        meds[ci] = float(np.median(tr.data))
    return meds


def pad_and_zscore(stream, wf_stt, wf_ent, pad_values, target_npts=9000, fs=100.0):
    """Slice the stream to [wf_stt, wf_ent], pad to target_npts using the
    PRE-P MEDIAN per channel (passed as `pad_values`), then per-channel demean
    + zscore. Returns (3, target_npts) float32."""
    expected_end = wf_stt + target_npts / fs
    sl = deepcopy(stream).slice(starttime=wf_stt, endtime=expected_end + 0.001)
    data = np.zeros((3, target_npts), dtype=np.float32)
    # initialize all channels with the per-channel pad value
    for ci in range(3):
        data[ci, :] = pad_values[ci]
    for ci in range(3):
        if ci >= len(sl):
            continue
        tr = sl[ci]
        offset_sec = float(tr.stats.starttime - wf_stt)
        offset_pt = int(round(offset_sec * fs))
        n = min(len(tr.data), target_npts - offset_pt)
        if n <= 0:
            continue
        lo = max(0, offset_pt)
        hi = min(target_npts, offset_pt + n)
        data[ci, lo:hi] = tr.data[: hi - lo]
    # per-channel demean + zscore (matches v49 training-time normalization)
    data -= data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    data = data / std
    return data


def stream_to_zraw(stream, wf_stt, wf_ent, pad_value_z, target_npts=9000, fs=100.0):
    """Raw Z (channel 2, max-abs scaled) for the polarity stream input."""
    sl = deepcopy(stream).slice(starttime=wf_stt, endtime=wf_stt + target_npts / fs + 0.001)
    z = np.full((1, target_npts), pad_value_z, dtype=np.float32)
    if len(sl) >= 3:
        tr = sl[2]
        offset_sec = float(tr.stats.starttime - wf_stt)
        offset_pt = int(round(offset_sec * fs))
        n = min(len(tr.data), target_npts - offset_pt)
        if n > 0:
            lo = max(0, offset_pt)
            hi = min(target_npts, offset_pt + n)
            z[0, lo:hi] = tr.data[: hi - lo]
    m = float(np.max(np.abs(z)))
    if m > 1e-6:
        z = z / m
    return z


def predict_batch_90s(model, data_norm, z_raw, device, batch_size=64, pred_npts=9000):
    """Forward over a batch of (3, T) windows. Returns (N, T) P-prob + (N, T) mask."""
    N = data_norm.shape[0]
    P_all = np.empty((N, pred_npts), dtype=np.float32)
    D_all = np.empty((N, pred_npts), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            x = torch.from_numpy(data_norm[i : i + batch_size]).to(device)
            zr = torch.from_numpy(z_raw[i : i + batch_size]).to(device)
            out = model.backbone.inner(x, z_raw=zr)
            picker = out[0]  # (B, 3, T) — channels [P, S, N]
            detector = out[-1]  # (B, 2, T) — channels [event, no-event]
            P_all[i : i + batch_size] = picker[:, 0, :].detach().cpu().numpy()
            D_all[i : i + batch_size] = detector[:, 0, :].detach().cpu().numpy()
    return P_all, D_all


def process_station_windows(P_func, M_func, wf_stt_arr, wf_ent_arr, gt_p_utc=None):
    """Peak extraction + delay logic on per-window predictions (90 s path).

    Returns: dict keyed by inf_id (str of window_end[:22]) -> {delay, P_probability, Mask_probability}.
    """
    out = {}
    for j in range(len(P_func)):
        inf_id = str(wf_ent_arr[j])[:22]
        out[inf_id] = {"delay": -1, "P_probability": -1, "Mask_probability": -1}

        rt_p = P_func[j]
        rt_M = M_func[j]
        search_range = trigger_onset(
            rt_p, TRIGGER_ON_OFF[0], TRIGGER_ON_OFF[1], max_len=TRIGGER_MAX_LEN
        )
        peaks, props = find_peaks(rt_p, height=P_DETECT_THRE, distance=int(1 / DT))
        if len(peaks) == 0:
            continue
        heights = props["peak_heights"]
        # keep only peaks that fall inside a trigger_onset region; trigger_onset's end index is
        # the last active sample (inclusive), so a peak on it is not kept (README, Known caveats)
        keep = np.zeros(len(peaks), dtype=bool)
        for k, p_idx in enumerate(peaks):
            for s in search_range:
                if s[0] <= p_idx < s[1]:
                    keep[k] = True
                    break
        peaks = peaks[keep]
        heights = heights[keep]
        if len(peaks) == 0:
            continue

        for trg_peak, trg_pb in zip(peaks, heights):
            mask_pb = float(np.mean(rt_M[trg_peak:]))
            trg_p_utc = wf_stt_arr[j] + trg_peak * DT
            delay = float(wf_ent_arr[j] - trg_p_utc)
            if gt_p_utc is not None and abs(trg_p_utc - gt_p_utc) > P_TRUE_PICK_THRE:
                continue
            if delay > DELAY_THRE:
                continue
            out[inf_id]["delay"] = delay
            out[inf_id]["P_probability"] = float(trg_pb)
            out[inf_id]["Mask_probability"] = mask_pb
    return out


# ──────────────────────────────────────────────────────────────────────────
# 60 s path: REDPAN_tools.data_utils (vendored verbatim)
# ──────────────────────────────────────────────────────────────────────────
def sac_len_complement(wf, max_length=None):
    """Complement sac data into the same length

    Vendored as is. Note: each pass of the inner loop appends ``len(data)`` zeros, so a trace
    short by k samples grows to ``len * 2**k`` samples (doubled for k = 1) rather than by k; the
    extra zeros lie after the record end. Kept for reproduction; a warning is logged when more
    than one sample is missing.
    """
    wf_n = np.array([len(i.data) for i in wf])
    if not max_length:
        max_n = np.max(wf_n)
    else:
        max_n = max_length

    append_wf_id = np.where(wf_n != max_n)[0]
    for w in append_wf_id:
        append_npts = max_n - len(wf[w].data)
        if append_npts > 1:
            log.warning(
                "sac_len_complement: %s is %d samples short; the vendored loop grows it "
                "to 2**%d times its length",
                wf[w].id,
                append_npts,
                append_npts,
            )
        if append_npts > 0:
            for p in range(append_npts):
                wf[w].data = np.insert(wf[w].data, len(wf[w].data), np.zeros(len(wf[w].data)))
        elif append_npts < 0:
            wf[w].data = wf[w].data[:max_n]
    return wf


def stream_standardize(st, data_length):
    """
    input: obspy.stream object (raw data)
    output: obspy.stream object (standardized)

    Vendored as is. Note: a short trace is padded with ``np.insert(x, -1, zeros)``, i.e. the
    zeros go before its last sample, so the data stay at the start of the window. In the 60 s
    replay a window that starts before the record start is therefore end-padded and shifted in
    time; run_streaming.py counts such windows. Kept for reproduction.
    """
    data_len = [len(i.data) for i in st]
    check_len = np.array_equal(data_len, np.repeat(data_length, 3))
    if not check_len:
        for s in st:
            res_len = len(s.data) - data_length
            if res_len > 0:
                s.data = s.data[:data_length]
            elif res_len < 0:
                last_pt = 0  # s.data[-1]
                s.data = np.insert(s.data, -1, np.repeat(last_pt, -res_len))

    st = st.detrend("demean")
    for s in st:
        data_std = np.std(s.data)
        if data_std == 0:
            data_std = 1
        s.data /= data_std
        s.data[np.isinf(s.data)] = 0
        s.data[np.isnan(s.data)] = 0
    return st


@torch.no_grad()
def predict_60s(model, wf, device, batch=256):
    """Channel-last (N, T, 3) windows -> (Pick (N, T, 3), Mask (N, T, 2)), the contract of the
    original TF model (the ``TF60Torch`` wrapper used for the manuscript replay)."""
    net = model.backbone.inner
    wf = np.asarray(wf, dtype=np.float32)
    if wf.ndim == 2:
        wf = wf[None]
    N = wf.shape[0]
    Pick = np.empty((N, wf.shape[1], 3), dtype=np.float32)
    Mask = np.empty((N, wf.shape[1], 2), dtype=np.float32)
    for i in range(0, N, batch):
        chunk = wf[i : i + batch].transpose(0, 2, 1)  # -> (b,3,T)
        x = torch.from_numpy(np.ascontiguousarray(chunk)).to(device)
        picker, detector = net(x)  # (b,3,T),(b,2,T)
        Pick[i : i + batch] = picker.permute(0, 2, 1).cpu().numpy()
        Mask[i : i + batch] = detector.permute(0, 2, 1).cpu().numpy()
    return Pick, Mask
