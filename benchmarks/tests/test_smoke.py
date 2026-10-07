"""Smoke test: every scorer runs end to end on a tiny synthetic results tree, and every entry
point answers --help. No real data and no checkpoints are needed.

    cd benchmarks && python -m pytest -q tests/
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BENCH = Path(__file__).resolve().parents[1]
ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(
        [str(BENCH), str(BENCH.parent), os.environ.get("PYTHONPATH", "")]
    ),
}
RNG = np.random.default_rng(0)
POOLS = ("STEAD_test", "GeoNet_test", "INSTANCE_test", "RockNet_test", "TW_test")
STATIC_SETS = (
    "ceed_nc",
    "ceed_sc",
    "crew",
    "geonet",
    "geonet_val",
    "instance",
    "romplus",
    "stead",
    "stead_h5",
    "tw",
)
STEAD_EQ = [f"ST{k:03d}.XX_2017_EV" for k in range(12)]
STEAD_NZ = [f"ST{k:03d}.XX_2017_NO" for k in range(6)]


def static_rows(evids, label, noise=False):
    rows = []
    for e in evids:
        p = float(RNG.uniform(5, 30))
        s = p + float(RNG.uniform(1, 40))
        for k in range(int(RNG.integers(1, 3))):
            on = p - 1 + k * 20
            rows.append(
                dict(
                    evid=e,
                    label_type="noise" if noise else "earthquake",
                    mode="static",
                    model=label,
                    labelP_sec=-999.0 if noise else p,
                    labelS_sec=-999.0 if noise else s,
                    ps_diff_sec=-999.0 if noise else s - p,
                    polarity_label="",
                    n_triggers=1,
                    trigger_idx=k,
                    trigger_on_sec=on,
                    trigger_off_sec=on + float(RNG.uniform(2, 45)),
                    mask_peak=float(RNG.uniform(0.3, 1)),
                    mask_mean=float(RNG.uniform(0.1, 1)),
                    P_pick_sec=p,
                    P_pick_prob=float(RNG.uniform(0, 1)),
                    S_pick_sec=s,
                    S_pick_prob=float(RNG.uniform(0, 1)),
                    polarity_at_P=np.nan,
                    P_residual_sec=np.nan if noise else float(RNG.normal(0, 0.3)),
                    S_residual_sec=np.nan if noise else float(RNG.normal(0, 0.6)),
                )
            )
    return rows


def make_tree(root: Path):
    res = root / "results"
    for m in ("edge", "rpm", "redpan"):
        for ds in STATIC_SETS:
            eq = STEAD_EQ if ds == "stead" else [f"{ds}_{i}" for i in range(12)]
            nz = STEAD_NZ if ds == "stead" else [f"{ds}_n{i}" for i in range(6)]
            rows = static_rows(eq, m) + (
                static_rows(nz, m, True) if ds in ("geonet", "instance", "tw", "stead") else []
            )
            (res / "static" / m).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(res / "static" / m / f"{ds}.csv", index=False)
        pd.DataFrame(static_rows(STEAD_NZ, m, True)).to_csv(
            res / "static" / m / "stead_noise_test.csv", index=False
        )
        (res / "noise" / m).mkdir(parents=True, exist_ok=True)
        for pool in POOLS:
            trig = [
                dict(
                    hdf5_index=i,
                    category="noise",
                    n_triggers=1,
                    trigger_idx=0,
                    trigger_on_sec=1.0,
                    trigger_off_sec=3.0,
                    mask_peak=float(RNG.uniform(0, 1)),
                    mask_mean=float(RNG.uniform(0, 1)),
                    inside_max_P=float(RNG.uniform(0, 1)),
                    inside_max_S=float(RNG.uniform(0, 1)),
                )
                for i in range(8)
            ]
            pd.DataFrame(trig).to_csv(
                res / "noise" / m / f"noisefp_triggers_{pool}.csv", index=False
            )
            pd.DataFrame(
                [dict(hdf5_index=i, det_trigger_max=float(RNG.uniform(0, 1))) for i in range(8)]
            ).to_csv(res / "noise" / m / f"noisefp_{pool}.csv", index=False)
    for b in ("phasenet_stead", "phasenet_instance", "eqt_stead", "eqt_instance"):
        (res / "native" / b).mkdir(parents=True, exist_ok=True)
        for ds in ("stead", "geonet", "crew", "tw", "instance", "romplus"):
            eq = STEAD_EQ if ds == "stead" else [f"{ds}_{i}" for i in range(12)]
            nz = STEAD_NZ if ds == "stead" else [f"{ds}_n{i}" for i in range(6)]
            rows = [dict(r, trigger_idx=0) for r in static_rows(eq, b) + static_rows(nz, b, True)]
            pd.DataFrame(rows).drop_duplicates("evid").to_csv(
                res / "native" / b / f"{ds}.csv", index=False
            )
        for ds in ("geonet_noise_test", "rocknet_noise_test"):
            pd.DataFrame(static_rows([f"{ds}_{i}" for i in range(5)], b, True)).drop_duplicates(
                "evid"
            ).to_csv(res / "native" / b / f"{ds}.csv", index=False)
    (res / "polarity").mkdir(parents=True, exist_ok=True)
    names = [f"nc{i}/XX.S{i}..HH" for i in range(40)]
    gt = RNG.choice(["U", "D", "N"], size=40)
    for m in ("edge", "rpm"):
        u = RNG.uniform(0, 1, 40)
        d = (1 - u) * RNG.uniform(0, 1, 40)
        pd.DataFrame(
            dict(trace_name=names, gt_polarity=gt, polarity_u_prob=u, polarity_d_prob=d)
        ).to_csv(res / "polarity" / f"{m}_polarity_per_event.csv", index=False)
    with open(BENCH / "data" / "eew_matched800_noise.txt") as fh:
        matched = fh.readline().strip()  # e.g. StrongMotion_00001.csv
    for m in ("edge", "rpm", "redpan"):
        for mode, fname in (("eq", "TSMIP_12345678.P20.csv"), ("noise", matched)):
            d = res / "streaming" / m / mode
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                dict(
                    inference_endtime=[
                        f"2020-01-01T00:00:{s:05.2f}" for s in np.arange(10, 12, 0.05)
                    ],
                    delay=RNG.uniform(-0.5, 2.5, 40),
                    P_probability=RNG.uniform(0, 1, 40),
                    Mask_probability=RNG.uniform(0, 1, 40),
                    station_id=["ST1"] * 40,
                )
            ).to_csv(d / fname, sep="\t", index=False)
    # STEAD membership files used by the held-out filter
    (root / "h5" / "STEAD").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(sample_id=[f"singleEQ_{e}" for e in STEAD_EQ[:3]], split="train")).to_csv(
        root / "h5" / "STEAD" / "STEAD_dataset_90s_singleEQ_metadata.csv", index=False
    )
    (root / "stead").mkdir(exist_ok=True)
    np.save(
        root / "stead" / "test.npy", np.array(STEAD_EQ + STEAD_NZ, dtype=object), allow_pickle=True
    )
    cfg = root / "cfg.yaml"
    cfg.write_text(
        f"paths:\n  results_root: {res}\n  h5_root: {root / 'h5'}\n  stead_root: {root / 'stead'}\n"
    )
    return cfg, res


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return make_tree(tmp_path_factory.mktemp("bench"))


def run(*args):
    r = subprocess.run(
        [sys.executable, *args], cwd=BENCH, env=ENV, capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.mark.parametrize(
    "script",
    [
        "score_table2.py",
        "score_table3.py",
        "score_table4.py",
        "score_fig3.py",
        "score_duration.py",
        "score_table5.py",
        "score_streaming.py",
    ],
)
def test_scorer_runs(tree, script, tmp_path):
    cfg, res = tree
    extra = (
        ["--out", str(tmp_path / "f1.csv")]
        if script == "score_fig3.py"
        else ["--out-json", str(tmp_path / "o.json")]
    )
    run(script, "--config", str(cfg), *extra)
    out = tmp_path / ("f1.csv" if script == "score_fig3.py" else "o.json")
    assert out.exists() and out.stat().st_size > 0
    if out.suffix == ".json":
        with open(out) as fh:
            json.load(fh)


@pytest.mark.parametrize(
    "script",
    [
        "run_static.py",
        "run_noise.py",
        "run_native.py",
        "run_polarity.py",
        "run_streaming.py",
        "int8_pick_f1.py",
        "build_stead_noise_test.py",
        "build_geonet_holdout.py",
        "plot_fig3.py",
    ]
    + [
        f"score_{s}.py"
        for s in ("table2", "table3", "table4", "fig3", "duration", "table5", "streaming")
    ],
)
def test_help(script):
    assert "usage" in run(script, "--help").lower()
