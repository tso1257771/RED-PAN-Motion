"""The native pick rule (rpm_bench/picks.py), the pick F1 and ``score_table3.py --pick-f1``.

    cd benchmarks && python -m pytest -q tests/test_picks.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from rpm_bench import native
from rpm_bench.constants import DT, SENTINEL
from rpm_bench.picks import best_peak_near, pick_f1, record_picks

from test_smoke import make_tree, run


def bump(n, centre, height, width=20):
    x = np.arange(n)
    return (height * np.exp(-0.5 * ((x - centre) / width) ** 2)).astype(np.float32)


def test_native_uses_the_shared_rule():
    assert native.best_peak_near is best_peak_near


def test_best_peak_near_window_and_record():
    prob = bump(3000, 1000, 0.4) + bump(3000, 1030, 0.0) + bump(3000, 2000, 0.9)
    assert best_peak_near(prob, 1020, 0.5)[0] == 1000  # within 50 samples of the label
    assert best_peak_near(prob, 1100, 0.5) == (-1, 0.0)  # no peak within 0.5 s
    s, h = best_peak_near(prob, None, 0.5)  # noise: highest peak of the record
    assert s == 2000 and h == pytest.approx(0.9, abs=1e-6)


def test_record_picks_earthquake_takes_the_peak_near_the_label():
    # P: a weak peak at the label and a stronger one 5 s later (another event) -> the weak one
    p = bump(9000, 900, 0.35) + bump(9000, 1400, 0.95)
    s = bump(9000, 2050, 0.6)
    info = dict(evid="e1", label_type="earthquake", labelP_sec=9.0, labelS_sec=20.0)
    r = record_picks(info, p, s)
    assert r["P_pick_sec"] == pytest.approx(9.0) and r["P_pick_prob"] == pytest.approx(
        0.35, abs=1e-6
    )
    assert r["S_residual_sec"] == pytest.approx(0.5)
    far = dict(info, labelS_sec=40.0)  # no S peak within 1 s of the label
    r2 = record_picks(far, p, s)
    assert (
        r2["S_pick_sec"] == SENTINEL and r2["S_pick_prob"] == 0.0 and np.isnan(r2["S_residual_sec"])
    )


def test_record_picks_noise_takes_the_highest_peak():
    p = bump(6000, 500, 0.2) + bump(6000, 4000, 0.45)
    info = dict(evid="n1", label_type="noise", labelP_sec=SENTINEL, labelS_sec=SENTINEL)
    r = record_picks(info, p, np.zeros(6000, np.float32))
    assert r["P_pick_sec"] == pytest.approx(4000 * DT) and r["P_pick_prob"] == pytest.approx(
        0.45, abs=1e-6
    )
    assert r["S_pick_prob"] == 0.0 and np.isnan(r["P_residual_sec"])


def test_pick_f1_counts():
    eq = pd.DataFrame(
        dict(
            P_pick_prob=[0.9, 0.31, 0.29, 0.0],
            P_residual_sec=[0.1, -0.4, 0.0, np.nan],
            S_pick_prob=[0.5, 0.5, 0.5, 0.5],
            S_residual_sec=[0.2, 0.9, 1.1, np.nan],
        )
    )
    nz = pd.DataFrame(dict(P_pick_prob=[0.5, 0.1], S_pick_prob=[0.0, 0.3]))
    p = pick_f1(eq, nz, "P", 0.3)
    assert (p["tp"], p["fp"], p["fn"]) == (2, 1, 2) and p["f1"] == pytest.approx(4 / 7)
    s = pick_f1(eq, nz, "S", 0.3)
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 2)
    none = pick_f1(eq, nz.iloc[:0], "P", 0.3)
    assert none["fp"] == 0 and none["n_noise"] == 0


def test_score_table3_pick_f1(tmp_path):
    cfg, res = make_tree(tmp_path)
    out = tmp_path / "t3.json"
    run("score_table3.py", "--config", str(cfg), "--pick-f1", "--out-json", str(out))
    with open(out) as fh:
        d = json.load(fh)
    for m in ("edge", "redpan", "phasenet_stead", "eqt_instance"):
        pk = d[m]["pick_f1"]
        assert {"macro_P", "macro_S"} <= set(pk)
        assert 0.0 <= pk["geonet"]["P"]["f1"] <= 1.0
    assert d["edge"]["pick_f1"]["crew"]["P"]["n_noise"] == 0  # the static CREW run has no noise
    # GeoNet noise: the test split by default, the holdout's noise with --geonet-noise holdout
    assert d["edge"]["pick_f1"]["geonet"]["P"]["n_noise"] == 5
    out2 = tmp_path / "t3h.json"
    run(
        "score_table3.py",
        "--config",
        str(cfg),
        "--geonet-noise",
        "holdout",
        "--out-json",
        str(out2),
    )
    with open(out2) as fh:
        assert json.load(fh)["edge"]["geonet"]["n_noise"] == 6
