"""Unit tests of pure functions of the harness: config resolution, window cutting, trigger
extraction and the streaming false-positive key. Synthetic inputs only.

    cd benchmarks && python -m pytest -q tests/
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import score_streaming
import torch
from rpm_bench import config
from rpm_bench.polarity import crop_to_window
from rpm_bench.static import _build_static_window_eq, accepts_z_raw, extract_triggers


@pytest.fixture
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("RPM_BENCH_"):
            monkeypatch.delenv(k)
    return monkeypatch


# ---------------------------------------------------------------- config
def test_config_precedence_defaults_yaml_env(tmp_path, clean_env):
    cfg = config.load()
    assert cfg["data_root"] == Path("../data")
    assert cfg["h5_root"] == Path("../data/input_h5_90sec_v3")
    y = tmp_path / "c.yaml"
    y.write_text("paths:\n  data_root: /yaml/data\n  h5_root: /yaml/h5\n")
    cfg = config.load(str(y))
    assert cfg["h5_root"] == Path("/yaml/h5")  # YAML over the default
    assert cfg["stead_root"] == Path("/yaml/data/STEAD")  # default value, YAML reference
    assert cfg["stead_noise_test_dir"] == Path("/yaml/h5/STEAD/STEAD_noise_test_90s")
    clean_env.setenv("RPM_BENCH_H5_ROOT", "/env/h5")
    cfg = config.load(str(y))
    assert cfg["h5_root"] == Path("/env/h5")  # environment over YAML
    assert cfg["stead_noise_test_dir"] == Path("/env/h5/STEAD/STEAD_noise_test_90s")
    clean_env.setenv("RPM_BENCH_CONFIG", str(y))  # YAML named by the environment
    assert config.load()["stead_root"] == Path("/yaml/data/STEAD")


def test_config_unknown_key_warns(tmp_path, clean_env, caplog):
    y = tmp_path / "c.yaml"
    y.write_text("paths:\n  base: /b\n  h5_rot: /typo\n  data_root: '{base}/data'\n")
    with caplog.at_level(logging.WARNING, logger="rpm_bench.config"):
        cfg = config.load(str(y))
    assert "h5_rot" in caplog.text and "base" in caplog.text
    assert cfg["data_root"] == Path("/b/data")  # unknown keys still resolve references
    assert cfg["h5_root"] == Path("/b/data/input_h5_90sec_v3")


def test_config_errors(tmp_path, clean_env):
    with pytest.raises(SystemExit, match="config file not found"):
        config.load(str(tmp_path / "missing.yaml"))
    y = tmp_path / "c.yaml"
    y.write_text("paths:\n  h5_root: '{nope}/h5'\n")
    with pytest.raises(ValueError, match="unknown key"):
        config.load(str(y))
    y.write_text("paths:\n  data_root: '{h5_root}/x'\n")  # data_root -> h5_root -> data_root
    with pytest.raises(ValueError, match="circular"):
        config.load(str(y))


# ---------------------------------------------------------------- windows
def test_crop_to_window():
    wf = np.arange(3 * 100, dtype=np.float32).reshape(3, 100)
    out, new_p, start = crop_to_window(wf, 50, 40)
    assert out.shape == (3, 40) and new_p == 20 and start == 30
    np.testing.assert_array_equal(out, wf[:, 30:70])
    out, new_p, start = crop_to_window(wf, 5, 40)  # leaves the record at the front
    assert new_p == 20 and start == -15
    assert (out[:, :15] == 0).all()
    np.testing.assert_array_equal(out[:, 15:], wf[:, :25])
    out, new_p, start = crop_to_window(wf, 95, 40)  # and at the end
    assert new_p == 20 and start == 75
    np.testing.assert_array_equal(out[:, :25], wf[:, 75:])
    assert (out[:, 25:] == 0).all()


@pytest.mark.parametrize("in_samples", [9000, 6000])
def test_static_window_places_p_at_10_percent(in_samples):
    np.random.seed(0)
    n = 20000
    wf = np.stack(
        [np.arange(n, dtype=np.float32) + c * 1e5 for c in range(3)], axis=1
    )  # value = sample index
    k = int(round(0.1 * in_samples))  # 900 or 600
    win, real_start, real_end, front = _build_static_window_eq(wf, 5000, 7000, in_samples)
    assert win.shape == (in_samples, 3) and front == 0
    assert win[k, 0] == 5000 and real_start == 5000 - k and real_end == real_start + in_samples
    win, real_start, real_end, front = _build_static_window_eq(
        wf, 300, 2000, in_samples
    )  # P near the start
    assert win.shape == (in_samples, 3) and front == k - 300 and real_start == 0
    assert win[k, 0] == 300
    win, real_start, real_end, front = _build_static_window_eq(
        wf, n - 100, n - 50, in_samples
    )  # near the end
    assert win.shape == (in_samples, 3) and real_end == n
    assert win[k, 0] == n - 100


# ---------------------------------------------------------------- triggers
def test_extract_triggers_small_array():
    n = 200
    mask = np.zeros(n, np.float32)
    mask[50:100] = 0.8
    p = np.zeros(n, np.float32)
    p[60] = 0.9
    p[120] = 0.95  # the second P peak is outside the trigger
    s = np.zeros(n, np.float32)
    s[90] = 0.7
    pol = np.linspace(0, 1, n).astype(np.float32)
    trig = extract_triggers(mask, p, s, pol, smooth_npts=1)
    assert len(trig) == 1
    t = trig[0]
    assert t["trigger_idx"] == 0
    assert t["trigger_on_sec"] == pytest.approx(0.50) and t["trigger_off_sec"] == pytest.approx(
        0.99
    )
    assert t["mask_peak"] == pytest.approx(0.8) and t["mask_mean"] == pytest.approx(0.8)
    assert t["P_pick_sec"] == pytest.approx(0.60) and t["P_pick_prob"] == pytest.approx(0.9)
    assert t["S_pick_sec"] == pytest.approx(0.90) and t["S_pick_prob"] == pytest.approx(0.7)
    assert t["polarity_at_P"] == pytest.approx(pol[60])
    # trigger_onset's end index (99) is inclusive but the slices are [lo:hi]: a peak there is not
    # seen (README, Known caveats)
    p_end = p.copy()
    p_end[99] = 1.0
    assert extract_triggers(mask, p_end, s, None, smooth_npts=1)[0]["P_pick_sec"] == pytest.approx(
        0.60
    )
    # default 10-sample smoothing: still one trigger; no mask, no trigger
    assert len(extract_triggers(mask, p, s, None)) == 1
    assert extract_triggers(np.zeros(n, np.float32), p, s, None) == []


def test_accepts_z_raw():
    class Plain(torch.nn.Module):
        def forward(self, x):
            return x

    class WithZ(torch.nn.Module):
        def forward(self, x, z_raw=None):
            return x

    assert not accepts_z_raw(Plain()) and accepts_z_raw(WithZ())


# ---------------------------------------------------------------- streaming false-positive key
TIMES = ["2020-01-01T00:00:10.05", "2020-01-01T00:00:10.10", "2020-01-01T00:00:10.15"]
DELAYS = [0.45, 0.50, 0.55]  # one pick at 00:00:09.60 seen by three consecutive windows


@pytest.mark.parametrize("unit", ["ns", "us"])
def test_streaming_epoch_seconds_unit_independent(unit):
    t = pd.to_datetime(pd.Series(TIMES)).astype(f"datetime64[{unit}]")
    sec = score_streaming.epoch_seconds(t)
    assert sec.iloc[0] == pytest.approx(1577836810.05)
    assert (sec - pd.Series(DELAYS)).round(2).nunique() == 1


@pytest.mark.parametrize("unit", ["ns", "us"])
def test_streaming_one_pick_three_windows_one_false_positive(unit, monkeypatch):
    to_datetime = pd.to_datetime
    monkeypatch.setattr(
        pd, "to_datetime", lambda x, *a, **k: to_datetime(x, *a, **k).astype(f"datetime64[{unit}]")
    )
    common = dict(network="StrongMotion", station_id="ST1", P_probability=0.9, Mask_probability=0.9)
    eq = pd.DataFrame(
        [
            dict(
                common,
                evid="00001",
                ID="TSMIP_ST1_00001",
                delay=0.5,
                inference_endtime="2020-01-01T00:00:00.50",
            )
        ]
    )
    nz = pd.DataFrame(
        [
            dict(common, evid="00002", ID="StrongMotion_ST1_00002", delay=d, inference_endtime=t)
            for t, d in zip(TIMES, DELAYS)
        ]
    )
    total, best, asap = score_streaming.score(eq, nz)
    assert total == 1
    assert best[5] == 1 and asap[5] == 1  # FP column: one trigger, not three
    assert best[0] == pytest.approx(2 / 3)
