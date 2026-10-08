"""SeisBench PhaseNet / EQTransformer in their native configuration (Table III, Fig. 3 baselines).

The logic is copied unchanged from the benchmark harness that produced the manuscript numbers
(``benchmark_seisbench_native.py``). Sliding windows at the model's own length with 50% overlap
over the whole record (last window aligned to the end), overlap-averaged outputs, each window
normalized by the model's ``annotate_batch_pre``. Picks: ``find_peaks`` (height >= 0.1, >= 0.5 s
apart); the scored pick is the highest peak inside the tolerance window around the label.
Detection: EQTransformer's detection head; PhaseNet has none, so max(P, S) is used.
"""

from __future__ import annotations

import numpy as np
import seisbench.models as sbm
import torch
from scipy.signal import find_peaks

from .constants import DT, P_TOL_SEC, S_TOL_SEC, SENTINEL
from .picks import MIN_PROB, PEAK_MIN_DIST, SR, best_peak_near  # noqa: F401  (re-exported)


@torch.no_grad()
def sliding_inference(model, wf_TC: np.ndarray, kind: str, device: str):
    """Sliding window inference using model.in_samples + 50% overlap.

    Returns:
      For PhaseNet: (p, s, n)  — softmax over [P, S, N]
      For EQT:      (p, s, det) — sigmoid heads
    """
    T = wf_TC.shape[0]
    target_T = model.in_samples
    wf_3T = wf_TC.T.astype(np.float32, copy=False)

    if T <= target_T:
        offsets = [0]
    else:
        stride = target_T // 2
        offsets = list(range(0, T - target_T + 1, stride))
        if offsets[-1] + target_T < T:
            offsets.append(T - target_T)

    a_acc = np.zeros(T, dtype=np.float32)
    b_acc = np.zeros(T, dtype=np.float32)
    c_acc = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)

    for off in offsets:
        end = min(off + target_T, T)
        win = wf_3T[:, off:end].copy()
        if win.shape[1] < target_T:
            win = np.pad(win, ((0, 0), (0, target_T - win.shape[1])))
        # Use the model's own annotate_batch_pre for preprocessing
        # (PhaseNet uses peak norm, EQT uses std norm — handled by model)
        x = torch.from_numpy(win[None]).to(device)
        x = model.annotate_batch_pre(x, {})
        out = model(x)

        if kind == "phasenet":
            # out: (1, 3, T) — already softmaxed in PhaseNet's forward
            arr = out[0].cpu().numpy()
            a_acc[off:end] += arr[0, : end - off]  # P
            b_acc[off:end] += arr[1, : end - off]  # S
            c_acc[off:end] += arr[2, : end - off]  # N
        elif kind == "eqt":
            det, p, s = out
            a_acc[off:end] += p[0, : end - off].cpu().numpy()
            b_acc[off:end] += s[0, : end - off].cpu().numpy()
            c_acc[off:end] += det[0, : end - off].cpu().numpy()
        counts[off:end] += 1.0

    mask = counts > 0
    for arr in (a_acc, b_acc, c_acc):
        arr[mask] /= counts[mask]
    return a_acc, b_acc, c_acc


def process_trace(model_name: str, model, wf_TC, info, device) -> dict:
    """One record -> one row in the static-run schema (see run_native.py)."""
    if model_name.startswith("PHASENET_SB"):
        p_prob, s_prob, n_prob = sliding_inference(model, wf_TC, "phasenet", device)
        det_arr = np.maximum(p_prob, s_prob)  # PhaseNet has no detector
    elif model_name.startswith("EQT_SB"):
        p_prob, s_prob, det_arr = sliding_inference(model, wf_TC, "eqt", device)
    else:
        raise ValueError(model_name)

    label_type = info["label_type"]
    if label_type == "earthquake":
        p_gt = int(round(info["labelP_sec"] / DT)) if info["labelP_sec"] >= 0 else None
        s_gt = int(round(info["labelS_sec"] / DT)) if info["labelS_sec"] >= 0 else None
    else:
        p_gt = s_gt = None

    p_pick, p_pick_prob = best_peak_near(p_prob, p_gt, P_TOL_SEC)
    s_pick, s_pick_prob = best_peak_near(s_prob, s_gt, S_TOL_SEC)

    # Detection: peak of det_arr if present (EQT), else max(P, S) via find_peaks
    det_peaks, det_info = find_peaks(det_arr, height=MIN_PROB, distance=PEAK_MIN_DIST)
    if len(det_peaks):
        max_eq_prob = float(det_info["peak_heights"].max())
    else:
        max_eq_prob = float(det_arr.max())

    p_res = (
        (p_pick * DT - info["labelP_sec"])
        if (p_pick >= 0 and info["labelP_sec"] >= 0)
        else float("nan")
    )
    s_res = (
        (s_pick * DT - info["labelS_sec"])
        if (s_pick >= 0 and info["labelS_sec"] >= 0)
        else float("nan")
    )

    return dict(
        evid=info["evid"],
        label_type=label_type,
        mode="static_native",
        model=model_name,
        labelP_sec=info["labelP_sec"],
        labelS_sec=info["labelS_sec"],
        ps_diff_sec=(
            (info["labelS_sec"] - info["labelP_sec"])
            if (info["labelP_sec"] >= 0 and info["labelS_sec"] >= 0)
            else SENTINEL
        ),
        polarity_label=info.get("polarity_label", ""),
        n_triggers=len(det_peaks),
        trigger_idx=0,
        trigger_on_sec=SENTINEL,
        trigger_off_sec=SENTINEL,
        mask_peak=max_eq_prob,
        mask_mean=max_eq_prob,
        P_pick_sec=p_pick * DT if p_pick >= 0 else SENTINEL,
        P_pick_prob=float(p_pick_prob),
        S_pick_sec=s_pick * DT if s_pick >= 0 else SENTINEL,
        S_pick_prob=float(s_pick_prob),
        polarity_at_P=float("nan"),  # neither model emits polarity here
        P_residual_sec=p_res,
        S_residual_sec=s_res,
    )


def load_model(model_spec: str, device: str):
    """model_spec: 'phasenet:stead' or 'eqt:stead'."""
    kind, tag = model_spec.split(":", 1)
    kind = kind.lower()
    if kind == "phasenet":
        m = sbm.PhaseNet.from_pretrained(tag)
        out_name = f"PHASENET_SB_{tag.upper()}"
    elif kind == "eqt":
        m = sbm.EQTransformer.from_pretrained(tag)
        out_name = f"EQT_SB_{tag.upper()}"
    else:
        raise ValueError(f"unknown kind {kind!r}")
    m.eval()
    m.to(device)
    return m, out_name
