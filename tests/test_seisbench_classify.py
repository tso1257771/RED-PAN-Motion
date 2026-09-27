"""Behavioural tests for the native pairing behind ``classify()``.

Synthetic annotations only. ``_joint_classify`` reads the annotation traces
and the model's table name, nothing else, so an untrained model built from
the constructor is enough: no checkpoint, no waveform archive.

Run:  python -m pytest tests/test_seisbench_classify.py
"""
import logging

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("seisbench", reason="needs the [seisbench] extra")

from obspy import Stream, Trace, UTCDateTime

from redpan_motion.integrations.seisbench import (
    RedpanSB60s,
    RedpanSB90s,
    _joint_classify,
)

DT = 0.01
T0 = UTCDateTime(2023, 3, 12, 17, 42, 42, 204000)


@pytest.fixture(scope="module")
def m60():
    return RedpanSB60s()


@pytest.fixture(scope="module")
def m90():
    return RedpanSB90s()


def _event(n=6000, p_val=0.6, s_val=0.5, tp=20.0, sp_sec=10.0, with_s=True, with_mask=True):
    """One clean event: a P peak, an S peak sp_sec later, a mask wrapping both."""
    t = np.arange(n) * DT
    ts = tp + sp_sec
    p = p_val * np.exp(-0.5 * ((t - tp) / 0.3) ** 2)
    s = s_val * np.exp(-0.5 * ((t - ts) / 0.5) ** 2) if with_s else np.zeros(n)
    m = np.where((t >= tp - 0.5) & (t <= ts + 1.5), 0.95, 0.0) if with_mask else np.zeros(n)
    return p, s, m


def _annotations(model, station="TEST", t0=T0, polarity=None, only=None, **event):
    """A Stream shaped like annotate()'s output for one instrument.

    polarity=(up, down) adds constant Polarity_N/U/D traces. only=(...) keeps
    just those labels, to build a deliberately incomplete set.
    """
    p, s, m = _event(**event)
    chans = {"P": p, "S": s, "Detection": m}
    if polarity is not None:
        up, dn = polarity
        chans.update({"Polarity_U": np.full(len(p), up), "Polarity_D": np.full(len(p), dn),
                      "Polarity_N": np.full(len(p), 1.0 - up - dn)})
    cls = model.__class__.__name__
    st = Stream()
    for label, data in chans.items():
        if only is not None and label not in only:
            continue
        tr = Trace(data.astype("float32"))
        tr.stats.sampling_rate = 1.0 / DT
        tr.stats.starttime = t0
        tr.stats.network, tr.stats.station, tr.stats.channel = "XX", station, f"{cls}_{label}"
        st.append(tr)
    return st


def _phases(picks):
    return [p.phase for p in picks]


# -- one event ----------------------------------------------------------------

def test_one_event_gives_one_p_s_pair(m60):
    picks = _joint_classify(m60, _annotations(m60), {}).picks
    assert _phases(picks) == ["P", "S"]
    assert {p.trace_id for p in picks} == {"XX.TEST."}
    assert abs(float(picks[0].peak_time - (T0 + 20.0))) <= DT
    assert abs(float(picks[1].peak_time - (T0 + 30.0))) <= DT
    for p in picks:
        assert p.start_time <= p.peak_time <= p.end_time
        assert 0.0 < p.peak_value <= 1.0


def test_a_p_without_an_s_is_not_a_pick(m60):
    assert _joint_classify(m60, _annotations(m60, with_s=False), {}).picks == []


def test_a_p_outside_any_detection_is_not_a_pick(m60):
    assert _joint_classify(m60, _annotations(m60, with_mask=False), {}).picks == []


# -- more than one instrument --------------------------------------------------

def test_two_stations_with_the_same_window_both_get_picks(m60):
    """The regression. Segments were keyed on (start, length) only, so two
    stations fetched for one event, which share both, collapsed to one and
    the first station's picks vanished without a message."""
    ann = _annotations(m60, station="AAA") + _annotations(m60, station="BBB")
    picks = _joint_classify(m60, ann, {}).picks
    assert len(picks) == 4
    by_station = {}
    for p in picks:
        by_station.setdefault(p.trace_id, []).append(p.phase)
    assert by_station == {"XX.AAA.": ["P", "S"], "XX.BBB.": ["P", "S"]}


def test_segments_of_one_station_at_different_times_are_separate(m60):
    ann = _annotations(m60, t0=T0) + _annotations(m60, t0=T0 + 600.0)
    picks = _joint_classify(m60, ann, {}).picks
    assert _phases(picks) == ["P", "S", "P", "S"]
    assert abs(float(picks[2].peak_time - (T0 + 620.0))) <= DT


# -- incomplete or damaged annotations -----------------------------------------

def test_a_p_segment_with_no_partner_is_reported_not_swallowed(m60, caplog):
    ann = _annotations(m60, only=("P",))
    with caplog.at_level(logging.WARNING, logger="redpan_motion.integrations.seisbench"):
        picks = _joint_classify(m60, ann, {}).picks
    assert picks == []
    assert "no matching S or Detection" in caplog.text


def test_nan_holes_away_from_the_event_change_nothing(m60):
    ann = _annotations(m60)
    for tr in ann:
        tr.data[5000:5200] = np.nan
    picks = _joint_classify(m60, ann, {}).picks
    assert _phases(picks) == ["P", "S"]
    assert all(np.isfinite(p.peak_value) for p in picks)


def test_nan_over_the_event_removes_it(m60):
    ann = _annotations(m60)
    for tr in ann:
        tr.data[1500:3500] = np.nan
    assert _joint_classify(m60, ann, {}).picks == []


# -- thresholds ------------------------------------------------------------------

def test_explicit_threshold_is_a_floor_in_adaptive_mode(m60):
    """With the table on, the bin sets the thresholds. A threshold the caller
    passes explicitly must still mean "at least this"."""
    ann = _annotations(m60)
    assert _phases(_joint_classify(m60, ann, {}).picks) == ["P", "S"]
    assert _joint_classify(m60, ann, {"P_threshold": 0.9}).picks == []
    assert _joint_classify(m60, ann, {"S_threshold": 0.6}).picks == []
    kept = _joint_classify(m60, ann, {"P_threshold": 0.5, "S_threshold": 0.4}).picks
    assert _phases(kept) == ["P", "S"]


def test_detection_threshold_gates_only_the_flat_path(m60):
    """Documented: with the table on, Detection_threshold is replaced by the
    table's mask-mean gate; with sp_adaptive=False it delineates the trigger."""
    ann = _annotations(m60)
    flat_strict = {"sp_adaptive": False, "Detection_threshold": 0.99}
    assert _joint_classify(m60, ann, flat_strict).picks == []
    assert _phases(_joint_classify(m60, ann, {"Detection_threshold": 0.99}).picks) == ["P", "S"]
    assert _phases(_joint_classify(m60, ann, {"sp_adaptive": False}).picks) == ["P", "S"]


def test_flat_path_applies_its_thresholds(m60):
    ann = _annotations(m60)
    assert _joint_classify(m60, ann, {"sp_adaptive": False, "P_threshold": 0.9}).picks == []
    assert _joint_classify(m60, ann, {"sp_adaptive": False, "S_threshold": 0.9}).picks == []


def test_min_sp_sec_drops_close_pairs(m60):
    ann = _annotations(m60, sp_sec=1.5)
    assert _phases(_joint_classify(m60, ann, {}).picks) == ["P", "S"]
    assert _joint_classify(m60, ann, {"min_sp_sec": 2.0}).picks == []


def test_defaults_are_read_from_the_registered_annotate_args():
    """The numbers live in _annotate_args, once. A model whose registration
    differs must classify by its own registration, not by a literal here."""
    m = RedpanSB60s()
    ann = _annotations(m)
    assert _phases(_joint_classify(m, ann, {"sp_adaptive": False}).picks) == ["P", "S"]
    m._annotate_args["P_threshold"] = ("Detection threshold for P", 0.9)
    assert _joint_classify(m, ann, {"sp_adaptive": False}).picks == []


# -- polarity --------------------------------------------------------------------

@pytest.mark.parametrize("up,down,expect", [(0.7, 0.2, "up"), (0.2, 0.7, "down")])
def test_p_picks_carry_polarity_on_the_ninety_second_model(m90, up, down, expect):
    picks = _joint_classify(m90, _annotations(m90, polarity=(up, down)), {}).picks
    assert _phases(picks) == ["P", "S"]
    p, s = picks
    assert (p.polarity, p.polarity_value) == (expect, pytest.approx(max(up, down)))
    assert s.polarity is None and s.polarity_value is None


def test_polarity_is_read_at_the_p_index(m90):
    """Not a constant: up before P, down at and after it, and the call is down."""
    ann = _annotations(m90, polarity=(0.8, 0.1))
    u = ann.select(channel="*Polarity_U")[0]
    d = ann.select(channel="*Polarity_D")[0]
    u.data[1990:], d.data[1990:] = 0.1, 0.8
    p = _joint_classify(m90, ann, {}).picks[0]
    assert p.phase == "P" and p.polarity == "down"


def test_sixty_second_model_never_reports_polarity(m60):
    picks = _joint_classify(m60, _annotations(m60, polarity=(0.7, 0.2)), {}).picks
    assert _phases(picks) == ["P", "S"]
    assert all(p.polarity is None for p in picks)
