"""Contract tests for redpan_motion.integrations.seisbench.

Synthetic input only. These check the wrapper's contract with SeisBench and two
faults found in earlier versions of it, neither of which needs a real earthquake:

  * The polarity stream is max-abs scaled, not merely demeaned. The head is
    trained on a vertical in [-1, 1] and raw counts of order 1e5 saturate it.
    The picker output is unchanged when this is wrong, so nothing else catches
    it: on a real window where the native head called the first motion up
    with high confidence, the wrapper returned exactly [1, 0, 0] at an
    identical P probability.
  * classify() uses the native mask-gated pairing rather than SeisBench's
    per-phase thresholding, so it emits P and S in pairs and never a lone
    phase.

Agreement against the native path on real records needs waveforms, which this
repository does not ship, so it is not tested here.

Run:  REDPAN_CKPTS=./checkpoints python -m pytest tests/test_seisbench_integration.py
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("seisbench", reason='needs the [seisbench] extra')

from redpan_motion.integrations.seisbench import (
    RedpanSB60s,
    RedpanSB90s,
    _run_around,
    _state_dict_from,
)

CKPTS = os.environ.get("REDPAN_CKPTS", "checkpoints")
DEV = "cpu"
RNG = np.random.default_rng(0)


def _build(variant):
    if not os.path.exists(f"{CKPTS}/{variant}"):
        pytest.skip(f"needs {CKPTS}/{variant}")
    cls = RedpanSB60s if variant == "redpan_60s" else RedpanSB90s
    return cls.from_redpan_checkpoint(f"{CKPTS}/{variant}", device=DEV)


def _counts(n, npts, offset=7e4, scale=1e5):
    """Raw-counts-like input: large amplitude on a large DC offset."""
    return torch.tensor((RNG.normal(size=(n, 3, npts)) * scale + offset).astype("float32"),
                        device=DEV)


# -- _run_around ------------------------------------------------------------

def test_run_around_spans_the_peak():
    assert _run_around(np.array([0., .1, .6, 1., .6, .1, 0.]), 3, 0.5) == (2, 4)


def test_run_around_clamps_at_the_edges():
    assert _run_around(np.ones(5), 0, 0.5) == (0, 4)


def test_run_around_on_a_lone_sample():
    assert _run_around(np.array([0., 1., 0.]), 1, 0.5) == (1, 1)


def test_run_around_never_excludes_its_own_index():
    """SeisBench asserts start_time <= peak_time <= end_time when building a Pick."""
    for _ in range(50):
        prob = RNG.random(200)
        i = int(prob.argmax())
        lo, hi = _run_around(prob, i, prob[i] / 2.0)
        assert lo <= i <= hi


# -- table wiring -----------------------------------------------------------

def test_sp_tables_resolve():
    from redpan_motion.sp_thresholds import resolve_table
    for cls in (RedpanSB60s, RedpanSB90s):
        assert set(cls.SP_TABLES) == set(cls.VARIANTS)
    for name in [*RedpanSB60s.SP_TABLES.values(), *RedpanSB90s.SP_TABLES.values()]:
        assert resolve_table(name) is not None


def test_the_two_ninety_second_variants_use_different_tables():
    assert (RedpanSB90s.SP_TABLES["redpan_motion"]
            != RedpanSB90s.SP_TABLES["edge_rp90"])


def test_sp_table_is_per_instance_on_the_ninety_second_class():
    a, b = _build("redpan_motion"), _build("edge_rp90")
    assert a.sp_table != b.sp_table


# -- the polarity input regression -----------------------------------------

@pytest.mark.parametrize("variant", ["redpan_motion", "edge_rp90"])
def test_polarity_channel_is_demeaned_and_max_abs_scaled(variant):
    m = _build(variant)
    pre = m.annotate_batch_pre(_counts(2, m.in_samples), {})
    assert pre.shape[1] == 4, "picker gets 3 channels, polarity head a 4th"
    z = pre[:, 3]
    assert float(z.abs().max()) == pytest.approx(1.0, abs=1e-5), "not max-abs scaled"
    assert float(z.mean().abs()) < 1e-5, "not demeaned"


@pytest.mark.parametrize("variant", ["redpan_motion", "edge_rp90"])
def test_polarity_head_is_not_saturated(variant):
    """The failure mode: an unscaled stream drove the softmax to exactly [1,0,0]."""
    m = _build(variant)
    with torch.no_grad():
        out = m(m.annotate_batch_pre(_counts(1, m.in_samples), {}))
    pol = out[0, 4:7].cpu().numpy()
    assert np.allclose(pol.sum(axis=0), 1.0, atol=1e-3), "polarity is not a softmax"
    assert pol.max() < 1.0 - 1e-6, "polarity saturated to a one-hot"


def test_the_picker_channels_are_z_scored_not_max_abs():
    """The other three channels keep the z-score the picker was trained on."""
    m = _build("redpan_motion")
    pre = m.annotate_batch_pre(_counts(2, m.in_samples), {})
    x = pre[:, :3]
    assert float(x.std()) == pytest.approx(1.0, abs=0.05)
    assert float(x.mean().abs()) < 1e-4


# -- the SeisBench contract -------------------------------------------------

@pytest.mark.parametrize("variant", ["redpan_60s", "redpan_motion", "edge_rp90"])
def test_forward_returns_one_channel_per_label(variant):
    m = _build(variant)
    with torch.no_grad():
        out = m(m.annotate_batch_pre(_counts(1, m.in_samples), {}))
    assert out.shape[1] == len(m.labels)


def test_sixty_second_model_has_no_polarity():
    m = _build("redpan_60s")
    assert m.has_polarity is False
    assert not any(label.startswith("Polarity") for label in m.labels)


def test_ninety_second_models_have_polarity():
    for v in ("redpan_motion", "edge_rp90"):
        m = _build(v)
        assert m.has_polarity is True
        assert [label for label in m.labels if label.startswith("Polarity")] == [
            "Polarity_N", "Polarity_U", "Polarity_D"]


def test_annotate_args_are_per_instance():
    """A class attribute here let a 90 s build rewrite a 60 s model's blinding."""
    a = _build("redpan_60s")
    b = _build("redpan_motion")
    assert a._annotate_args is not b._annotate_args
    assert a._annotate_args["blinding"][1] == (a.in_samples // 12,) * 2
    assert b._annotate_args["blinding"][1] == (b.in_samples // 12,) * 2


def test_overlap_is_at_least_twice_blinding():
    """Otherwise the annotation comes back with a periodic hole."""
    for v in ("redpan_60s", "redpan_motion", "edge_rp90"):
        m = _build(v)
        blind = m._annotate_args["blinding"][1][0]
        assert m._annotate_args["overlap"][1] >= 2 * blind


def test_classify_kwargs_are_registered_with_seisbench(caplog):
    """Unregistered keys make annotate log 'Unknown argument ... will be ignored'.

    classify() forwards its kwargs to annotate, whose check is _verify_argdict,
    so that is what is exercised here: the keys must pass it without a warning
    on the seisbench logger. An untrained model is enough, the check reads
    only _annotate_args.
    """
    import logging
    for m in (RedpanSB60s(), RedpanSB90s()):
        with caplog.at_level(logging.WARNING, logger="seisbench"):
            m._verify_argdict({"sp_adaptive": False, "min_sp_sec": 2.0,
                               "P_threshold": 0.5, "S_threshold": 0.4,
                               "Detection_threshold": 0.6})
        assert "Unknown argument" not in caplog.text, caplog.text


def test_model_args_round_trip_on_both_classes():
    """SeisBench's save() writes get_model_args() and load() calls cls(**args),
    so every constructor argument that changes behaviour has to come back.
    highpass_freq was missing from the 60 s class."""
    a = RedpanSB60s(in_samples=3000, highpass_freq=2.0)
    args = a.get_model_args()
    assert args["highpass_freq"] == 2.0 and args["in_samples"] == 3000
    b = RedpanSB60s(**args)
    assert (b.highpass_freq, b.in_samples) == (2.0, 3000)

    a = RedpanSB90s(variant="edge_rp90", in_samples=4500, highpass_freq=0.5)
    args = a.get_model_args()
    b = RedpanSB90s(**args)
    assert (b.variant, b.in_samples, b.highpass_freq) == ("edge_rp90", 4500, 0.5)


def test_state_dict_from_accepts_the_three_checkpoint_layouts():
    bare = {"w": 1}
    assert _state_dict_from(bare) is bare
    assert _state_dict_from({"model_state_dict": bare, "epoch": 3}) is bare
    assert _state_dict_from({"state_dict": bare}) is bare
    assert _state_dict_from(None) is None


# -- review follow-up: one base class, save/load, normalization -----------------

@pytest.mark.parametrize("variant", ["redpan_60s", "redpan_motion", "edge_rp90"])
def test_seisbench_save_and_load_round_trip(variant, tmp_path):
    """SeisBench load() rebuilds through cls(**get_model_args()). model_kwargs
    was missing from those args, so redpan_motion, whose config sets a filter
    count the builder does not default to, failed with a size mismatch."""
    m = _build(variant)
    m.save(str(tmp_path / variant))
    r = type(m).load(str(tmp_path / variant))
    assert r.variant == m.variant and r.model_kwargs == m.model_kwargs
    r.eval()            # load() returns training mode; annotate() switches itself
    x = _counts(1, m.in_samples)
    with torch.no_grad():
        a = m(m.annotate_batch_pre(x.clone(), {}))
        b = r(r.annotate_batch_pre(x.clone(), {}))
    assert torch.equal(a, b)


def test_identity_reads_the_same_way_on_both_classes():
    for m in (_build("redpan_60s"), _build("redpan_motion"), _build("edge_rp90")):
        assert m.sp_table == type(m).SP_TABLES[m.variant]
        assert m.has_polarity == any(label.startswith("Polarity_") for label in m.labels)
    assert RedpanSB60s().variant == "redpan_60s"


def test_zscore_matches_the_native_floor_and_population_std():
    from redpan_motion.integrations.seisbench import _RedpanSBBase
    x = torch.tensor(RNG.normal(size=(1, 3, 9000)).astype("float32"))
    x[0, 0] = 5.0 + 1e-10 * torch.randn(9000)   # near-dead: std far below 1e-8
    z = _RedpanSBBase._zscore(x)
    assert float(z[0, 0].abs().max()) < 1e-6, "a near-dead channel was amplified"
    live = x[0, 1] - x[0, 1].mean()
    expected = live / live.std(correction=0)
    assert torch.allclose(z[0, 1], expected, atol=1e-6), "not the population std"


def test_a_checkpoint_directory_can_be_named_anything_if_variant_is_given(tmp_path):
    import shutil
    src = f"{CKPTS}/edge_rp90"
    if not os.path.exists(src):
        pytest.skip(f"needs {src}")
    dst = tmp_path / "some_copy"
    shutil.copytree(src, dst)
    m = RedpanSB90s.from_redpan_checkpoint(dst, device=DEV, variant="edge_rp90")
    assert m.variant == "edge_rp90"
    with pytest.raises(ValueError, match="pass variant="):
        RedpanSB90s.from_redpan_checkpoint(dst, device=DEV)


def test_the_sixty_second_wrapper_accepts_any_directory_name(tmp_path):
    import shutil
    src = f"{CKPTS}/redpan_60s"
    if not os.path.exists(src):
        pytest.skip(f"needs {src}")
    dst = tmp_path / "renamed"
    shutil.copytree(src, dst)
    assert RedpanSB60s.from_redpan_checkpoint(dst, device=DEV).variant == "redpan_60s"


def test_an_unknown_variant_is_refused():
    with pytest.raises(ValueError, match="variant must be one of"):
        RedpanSB90s(variant="redpan_60s")
