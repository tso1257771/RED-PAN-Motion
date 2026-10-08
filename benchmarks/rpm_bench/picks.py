"""The pick rule of the native SeisBench runs, shared with the RED-PAN pick rows and pick F1.

Scored pick: ``find_peaks`` on the per-sample probability (height >= 0.1, peaks >= 0.5 s apart);
on an earthquake record the highest peak within the tolerance window around the label (0.5 s for
P, 1.0 s for S), on a noise record the highest peak of the record.

Pick F1 at a threshold (Table III, ``score_table3.py --pick-f1``): an earthquake record is a true
positive if its scored pick has probability >= threshold (it lies within the tolerance by
construction), otherwise a false negative; a noise record is a false positive if its scored pick has
probability >= threshold. F1 = 2 TP / (2 TP + FP + FN), with no detection gate.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .constants import DT, P_TOL_SEC, S_TOL_SEC, SENTINEL

SR = 100
MIN_PROB = 0.1
PEAK_MIN_DIST = 50  # 0.5 s


def best_peak_near(
    prob: np.ndarray,
    gt_sample: int | None,
    tol_sec: float,
    sr: int = SR,
    min_height: float = MIN_PROB,
    min_distance: int = PEAK_MIN_DIST,
):
    """(sample, height) of the highest ``find_peaks`` peak within ``tol_sec`` of ``gt_sample``
    (of the highest peak overall when ``gt_sample`` is None); ``(-1, 0.0)`` if there is none."""
    peaks, info = find_peaks(prob, height=min_height, distance=min_distance)
    heights = info.get("peak_heights", np.zeros(len(peaks)))
    if len(peaks) == 0:
        return -1, 0.0
    if gt_sample is None:
        idx = int(np.argmax(heights))
        return int(peaks[idx]), float(heights[idx])
    tol = int(tol_sec * sr)
    within = np.abs(peaks - gt_sample) <= tol
    if not np.any(within):
        return -1, 0.0
    cand = np.where(within)[0]
    best = cand[int(np.argmax(heights[cand]))]
    return int(peaks[best]), float(heights[best])


def record_picks(info: dict, p_prob: np.ndarray, s_prob: np.ndarray) -> dict:
    """One row of scored P and S picks for a record, in the columns of the native rows
    (``P_pick_sec``, ``P_pick_prob``, ``P_residual_sec`` and the same for S). ``p_prob`` and
    ``s_prob`` are indexed like the record (sample 0 = record start), so the labels in ``info``
    (seconds from the record start; negative = none) give the label samples."""
    eq = info["label_type"] == "earthquake"
    lab_p, lab_s = info["labelP_sec"], info["labelS_sec"]
    p_gt = int(round(lab_p / DT)) if eq and lab_p >= 0 else None
    s_gt = int(round(lab_s / DT)) if eq and lab_s >= 0 else None
    p_pick, p_h = best_peak_near(p_prob, p_gt, P_TOL_SEC)
    s_pick, s_h = best_peak_near(s_prob, s_gt, S_TOL_SEC)
    return dict(
        evid=info["evid"],
        label_type=info["label_type"],
        labelP_sec=lab_p,
        labelS_sec=lab_s,
        P_pick_sec=p_pick * DT if p_pick >= 0 else SENTINEL,
        P_pick_prob=float(p_h),
        S_pick_sec=s_pick * DT if s_pick >= 0 else SENTINEL,
        S_pick_prob=float(s_h),
        P_residual_sec=(p_pick * DT - lab_p) if (p_pick >= 0 and lab_p >= 0) else float("nan"),
        S_residual_sec=(s_pick * DT - lab_s) if (s_pick >= 0 and lab_s >= 0) else float("nan"),
    )


def pick_f1(eq, nz, phase: str, thr: float) -> dict:
    """P or S pick F1 of one dataset from one row per record (``eq``: earthquakes, ``nz``: noise)."""
    tol = P_TOL_SEC if phase == "P" else S_TOL_SEC
    prob, res = eq[f"{phase}_pick_prob"], eq[f"{phase}_residual_sec"]
    tp = int(((prob >= thr) & (res.abs() <= tol)).sum())
    fn = len(eq) - tp
    fp = int((nz[f"{phase}_pick_prob"] >= thr).sum()) if len(nz) else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return dict(f1=f1, tp=tp, fp=fp, fn=fn, n_eq=len(eq), n_noise=len(nz))
