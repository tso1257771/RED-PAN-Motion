"""Tests for the flat-region guard in ``REDPANPredictor._build_flat_mask``.

The guard zeroes predictions wherever the raw input is constant, which stops
false triggers on the zero pads at a trace boundary. It scanned every channel,
so a channel that was constant for the whole trace flagged every window and the
prediction came back zero everywhere.

That is not a corner case. A caller that supplies three components fills an
absent horizontal with zeros to give the model the shape it expects, and such a
record still carries a usable vertical. Before the fix, peak P on a real
record fell to exactly zero when E was zero filled, in every inference mode
and under both normalisations.
"""
import numpy as np
import pytest

from redpan_motion.inference import REDPANPredictor

RNG = np.random.default_rng(0)


def _noise(n_ch=3, n=3000):
    return RNG.normal(size=(n_ch, n)).astype(np.float32)


def test_live_channels_are_not_flagged():
    wf = _noise()
    assert not REDPANPredictor._build_flat_mask(wf).any()


def test_boundary_pad_is_still_flagged():
    wf = _noise()
    wf[:, :400] = 0.0
    m = REDPANPredictor._build_flat_mask(wf)
    assert m[:400].all()
    assert not m[400:].any()


@pytest.mark.parametrize("dead", [0, 1, 2])
def test_one_constant_channel_does_not_flag_the_trace(dead):
    """The regression. A wholly constant channel is a filled channel."""
    wf = _noise()
    wf[dead] = 0.0
    assert not REDPANPredictor._build_flat_mask(wf).any()


def test_constant_channel_and_a_real_pad_still_flags_the_pad():
    wf = _noise()
    wf[0] = 0.0            # filled horizontal
    wf[:, -300:] = 0.0     # genuine boundary pad
    m = REDPANPredictor._build_flat_mask(wf)
    assert m[-300:].all()
    assert not m[:-300].any()


def test_all_channels_constant_is_still_all_flat():
    assert REDPANPredictor._build_flat_mask(np.zeros((3, 1000), np.float32)).all()


def test_nonzero_constant_channel_is_also_skipped():
    """A filled channel need not be zero. Any single repeated value counts."""
    wf = _noise()
    wf[1] = 7.5
    assert not REDPANPredictor._build_flat_mask(wf).any()
