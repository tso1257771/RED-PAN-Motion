"""The rolling-window default is per checkpoint, not one value for all three.

``long_trace_norm="rolling"`` divides each sample by a centred moving std, so
pre-event noise is lifted to unit amplitude. That is the point of it, but on a
low-SNR distant record it also flattens the onset, and EdgeRP90 at 309k
parameters then returns nothing where the larger checkpoints only lose
contrast. On a set of distant records, EdgeRP90 found the P on clearly more
of them with a 30000-sample window than at ``pred_npts``; see the note beside
``_ROLLING_WINDOW_BY_MODEL``.
"""
import os

import numpy as np
import pytest

from redpan_motion.checkpoints import CHECKPOINT_DIR
from redpan_motion.inference import REDPANPredictor
from redpan_motion.inference.predictor import _ROLLING_WINDOW_BY_MODEL

CKPTS = os.environ.get("REDPAN_CKPTS", str(CHECKPOINT_DIR))


def _have(variant):
    return os.path.exists(f"{CKPTS}/{variant}/best.pt")


def test_table_names_a_real_model_class():
    """The lookup is by class name, so a renamed class silently loses its entry."""
    from torch import nn

    from redpan_motion import models as rpm
    for cls_name in _ROLLING_WINDOW_BY_MODEL:
        cls = getattr(rpm, cls_name, None)
        assert isinstance(cls, type) and issubclass(cls, nn.Module), cls_name


@pytest.mark.skipif(not _have("edge_rp90"), reason="needs checkpoints/edge_rp90")
def test_edge_rp90_uses_the_wider_window():
    p = REDPANPredictor.from_checkpoint(f"{CKPTS}/edge_rp90/best.pt", device="cpu")
    assert type(p.model).__name__ == "EdgeRP90"
    assert p.rolling_window == 30000
    assert p.rolling_window != p.pred_npts


@pytest.mark.parametrize("variant", ["redpan_60s", "redpan_motion"])
def test_other_checkpoints_keep_pred_npts(variant):
    if not _have(variant):
        pytest.skip(f"needs checkpoints/{variant}")
    p = REDPANPredictor.from_checkpoint(f"{CKPTS}/{variant}/best.pt", device="cpu")
    assert p.rolling_window == p.pred_npts


@pytest.mark.skipif(not _have("edge_rp90"), reason="needs checkpoints/edge_rp90")
def test_a_wider_window_than_the_trace_still_runs():
    """A 60 s record is shorter than the 300 s window. uniform_filter1d
    reflects, so the result approaches a global z-score rather than failing."""
    p = REDPANPredictor.from_checkpoint(f"{CKPTS}/edge_rp90/best.pt", device="cpu")
    rng = np.random.default_rng(0)
    wf = rng.normal(size=(3, 6000)).astype(np.float32)
    picker, detector, _ = p.predict_arrays(wf, mode="single", postprocess=False)
    assert picker.shape == (6000, 3)
    assert np.isfinite(picker).all()
    assert np.isfinite(detector).all()


def test_a_wrapped_model_still_resolves_its_window():
    """The lookup keys on the class name. A DataParallel or torch.compile
    wrapper reports its own class, so it must be unwrapped first, as the
    CH_Z / z_raw detection in __init__ already does. Before this, a wrapped
    EdgeRP90 silently fell back to pred_npts and sp_adaptive_thresholds=True
    raised KeyError on the wrapper's name."""
    from torch import nn

    from redpan_motion.models import build_edge_rp90

    class Wrapped(nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.module = mod

        def forward(self, *a, **k):
            return self.module(*a, **k)

    model = build_edge_rp90()
    bare = REDPANPredictor(model=model, pred_npts=9000, device="cpu")
    wrapped = REDPANPredictor(model=Wrapped(model), pred_npts=9000, device="cpu",
                              sp_adaptive_thresholds=True)
    assert bare.rolling_window == _ROLLING_WINDOW_BY_MODEL["EdgeRP90"]
    assert wrapped.rolling_window == bare.rolling_window
    assert wrapped.sp_thresholds is not None
