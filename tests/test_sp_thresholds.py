"""S-P-time-adaptive joint detection thresholds: table, duration mapping, and the
adaptive paths in detect_events_joint + REDPANPredictor._postprocess_threshold."""
import numpy as np
import pytest

from redpan_motion.sp_thresholds import (
    available_models, resolve_table, params_for_duration, default_params,
    floor_thresholds, sp_bin_from_duration, sp_bin_label,
)
from redpan_motion.picks import detect_events_joint


# ---------- table / mapping ----------
def test_available_and_resolve():
    assert set(available_models()) == {"edge_rp90", "rp90_motion_v49", "redpan_60s"}
    assert resolve_table(None) is None and resolve_table(False) is None
    assert resolve_table("edge_rp90")["default"] == (0.80, 0.20, 0.20)
    # class-name / model_type aliases (case-insensitive)
    assert resolve_table("EdgeRP90") is resolve_table("edge_rp90")
    assert resolve_table("Redpan60s") is resolve_table("redpan_60s")
    assert resolve_table("MTAN_R2UNet_RP90_Motion") is resolve_table("rp90_motion_v49")


def test_resolve_errors_and_custom_dict():
    with pytest.raises(KeyError):
        resolve_table("no_such_model")
    with pytest.raises(ValueError):
        resolve_table({"default": (0.9, 0.3, 0.2)})           # missing 'bins'
    custom = {"default": (0.5, 0.2, 0.1),
              "bins": [(0.6, 0.3, 0.2)] * 5}
    assert resolve_table(custom) is custom                     # passed through


def test_duration_binning():
    assert [sp_bin_from_duration(d) for d in (2, 5.9, 6, 11, 12, 20, 30)] == [0, 0, 1, 1, 2, 3, 4]
    assert sp_bin_label(3) == "[0,5)" and sp_bin_label(30) == "[20,+)"


def test_params_by_duration_and_floor():
    t = resolve_table("rp90_motion_v49")
    assert params_for_duration(3, t) == (0.60, 0.40, 0.20)     # near bin: low mask-mean, high P
    assert params_for_duration(30, t) == (0.90, 0.10, 0.10)    # far bin: high mask-mean, low P
    # floor = min over all bins+default; used so candidate peaks aren't pre-filtered
    m, p, s = floor_thresholds(t)
    assert p == 0.10 and s == 0.10 and m == 0.60


# ---------- detect_events_joint adaptive path ----------
def _trace(npts=9000):
    return (np.zeros(npts), np.zeros(npts), np.zeros(npts))    # p, s, mask


def test_adaptive_backward_compat_none():
    p, s, mask = _trace()
    mask[1000:1500] = 0.95; p[1010] = 0.9; s[1450] = 0.8
    assert detect_events_joint(p, s, mask, 0.01, 0.3, 0.2, 0.5) == \
           detect_events_joint(p, s, mask, 0.01, 0.3, 0.2, 0.5, sp_thresholds=None)


def test_adaptive_drops_pick_below_bin_threshold():
    # near event (~5 s trigger -> [0,5)); v49 near-bin P=0.40. A P peak of 0.35 passes the
    # FIXED floor (0.30) but must be dropped by the adaptive bin threshold (0.40). The strong
    # mask (mean ~0.94) clears the near-bin mask-mean gate (0.60), isolating the P drop.
    p, s, mask = _trace()
    mask[1000:1500] = 0.95; p[1010] = 0.35; s[1450] = 0.85
    fixed = detect_events_joint(p, s, mask, 0.01, 0.30, 0.20, 0.50)
    adapt = detect_events_joint(p, s, mask, 0.01, 0.30, 0.20, 0.50,
                                sp_thresholds=resolve_table("rp90_motion_v49"))
    assert len(fixed) == 1 and len(adapt) == 0


def test_adaptive_mask_mean_gate():
    # a sustained but WEAK trigger (mask 0.55, ~10 s -> [5,10)) fires at det_thr=0.5, but its
    # mask MEAN (~0.55) is below edge's [5,10) mean-mask gate (0.70) -> adaptive drops it.
    # (A peak gate would have passed it — 0.55 is the peak too — so this exercises MEAN.)
    p, s, mask = _trace()
    mask[1000:2000] = 0.55; p[1010] = 0.9; s[1950] = 0.85
    fixed = detect_events_joint(p, s, mask, 0.01, 0.30, 0.20, 0.50)
    adapt = detect_events_joint(p, s, mask, 0.01, 0.30, 0.20, 0.50,
                                sp_thresholds=resolve_table("edge_rp90"))
    assert len(fixed) == 1 and len(adapt) == 0


def test_adaptive_accepts_strong_event():
    # strong sustained trigger: mask mean ~0.94 and P/S well above edge's [0,5) bin gates
    # (0.60 / 0.20 / 0.20) -> adaptive keeps the pair.
    p, s, mask = _trace()
    mask[1000:1500] = 0.95; p[1010] = 0.9; s[1450] = 0.85
    adapt = detect_events_joint(p, s, mask, 0.01, 0.30, 0.20, 0.50,
                                sp_thresholds=resolve_table("edge_rp90"))
    assert len(adapt) == 1


# ---------- predictor wiring ----------
def test_predictor_auto_resolve_and_off():
    from redpan_motion.models import build_edge_rp90
    from redpan_motion.inference.predictor import REDPANPredictor
    m = build_edge_rp90().eval()
    assert REDPANPredictor(model=m, sp_adaptive_thresholds=True).sp_thresholds["default"] == (0.80, 0.20, 0.20)
    assert REDPANPredictor(model=m).sp_thresholds is None
    assert REDPANPredictor(model=m, sp_adaptive_thresholds="redpan_60s").sp_thresholds \
        is resolve_table("redpan_60s")


def test_predictor_postprocess_adaptive_gate():
    from redpan_motion.models import build_edge_rp90
    from redpan_motion.inference.predictor import REDPANPredictor
    m = build_edge_rp90().eval()
    T = 9000
    picker = np.zeros((T, 3)); detector = np.zeros((T, 2))
    detector[1000:1500, 0] = 0.50                             # mask mean 0.50
    picker[1010, 0] = 0.9; picker[1450, 1] = 0.85
    pr_fix = REDPANPredictor(model=m, dt=0.01)
    pr_adp = REDPANPredictor(model=m, dt=0.01, sp_adaptive_thresholds="edge_rp90")
    pr_fix._postprocess_threshold(picker, detector, 0.5, 0.3, 0.2, 0.1, 1)
    pr_adp._postprocess_threshold(picker, detector, 0.5, 0.3, 0.2, 0.1, 1)
    assert len(pr_fix._last_events) == 1        # mean-mask 0.50 >= fixed gate 0.50 accepts
    assert len(pr_adp._last_events) == 0        # mean-mask 0.50 < [0,5) bin gate 0.60 rejects
