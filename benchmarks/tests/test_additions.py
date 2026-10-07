"""Smoke tests for the GeoNet test-noise scorer, the streaming-window check and the archive builder,
on synthetic inputs (no real data, no checkpoints).

    cd benchmarks && python -m pytest -q tests/test_additions.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from test_smoke import BENCH, ENV, make_tree, static_rows

NEW_SCRIPTS = [
    "run_static_noise.py",
    "run_streaming_shard.py",
    "score_table3_geonet_test_noise.py",
    "check_streaming_windows.py",
]


def run(*args, env=None):
    r = subprocess.run(
        [sys.executable, *args], cwd=BENCH, env=env or ENV, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.mark.parametrize("script", NEW_SCRIPTS)
def test_help(script):
    assert "usage" in run(script, "--help").lower()


def test_builder_help_and_list(tmp_path):
    assert "usage" in run("-m", "builder", "--help").lower()
    out = run("-m", "builder", "--list", env={**ENV, "RPM_BENCH_DATA_ROOT": str(tmp_path)})
    for ds in (
        "STEAD",
        "INSTANCE",
        "GeoNet",
        "CREW",
        "OBSTransformer",
        "RockNet",
        "ROMPLUS",
        "CEED_NC",
        "CEED_SC",
        "TW",
    ):
        assert ds in out


def test_score_table3_geonet_test_noise(tmp_path):
    cfg, res = make_tree(tmp_path)
    for m in ("edge", "rpm", "redpan"):
        pd.DataFrame(static_rows([f"gtn_{i}" for i in range(7)], m, True)).to_csv(
            res / "static" / m / "geonet_test_noise.csv", index=False
        )
    out = tmp_path / "t3g.json"
    run("score_table3_geonet_test_noise.py", "--config", str(cfg), "--out-json", str(out))
    d = json.load(open(out))
    assert set(d["scores"]) == {
        "edge",
        "rpm",
        "redpan",
        "phasenet_stead",
        "phasenet_instance",
        "eqt_stead",
        "eqt_instance",
    }
    assert d["scores"]["edge"]["geonet_test_noise"]["n_noise"] == 7


def _fake_stead(root: Path, n_eq=8, n_noise=6):
    rng = np.random.default_rng(1)
    rows = []
    with h5py.File(root / "merge.hdf5", "w") as hf:
        g = hf.create_group("data")
        for k in range(n_eq):
            name = f"S{k:02d}.XX_20170101_EV"
            g.create_dataset(name, data=rng.normal(0, 1, (6000, 3)).astype(np.float32))
            rows.append(
                dict(
                    trace_name=name,
                    trace_category="earthquake_local",
                    source_id=f"ev{k // 2}",
                    p_arrival_sample=1000 + 50 * k,
                    s_arrival_sample=2000 + 80 * k,
                )
            )
        for k in range(n_noise):
            name = f"S{k:02d}.XX_20170101_NO"
            g.create_dataset(name, data=rng.normal(0, 1, (6000, 3)).astype(np.float32))
            rows.append(
                dict(
                    trace_name=name,
                    trace_category="noise",
                    source_id="",
                    p_arrival_sample=np.nan,
                    s_arrival_sample=np.nan,
                )
            )
    pd.DataFrame(rows).to_csv(root / "merge.csv", index=False)
    test = [r["trace_name"] for r in rows if r["trace_category"] == "noise"][:2]
    np.save(root / "test.npy", np.array(test, dtype=object), allow_pickle=True)
    return test


def test_builder_stead(tmp_path):
    stead = tmp_path / "STEAD"
    stead.mkdir()
    excluded = _fake_stead(stead)
    env = {**ENV, "RPM_BENCH_STEAD_ROOT": str(stead), "RPM_BENCH_BUILD_ROOT": str(tmp_path / "out")}
    run("-m", "builder", "--dataset", "STEAD", "--category", "noise", "--validate", env=env)
    run("-m", "builder", "--dataset", "STEAD", "--category", "singleEQ", "--validate", env=env)
    nz = pd.read_csv(tmp_path / "out/STEAD/STEAD_dataset_90s_noise_metadata.csv")
    assert len(nz) == 4 and not nz.sample_id.str.removeprefix("noise_").isin(excluded).any()
    assert set(nz.split) <= {"train", "val"}
    eq = pd.read_csv(tmp_path / "out/STEAD/STEAD_dataset_90s_singleEQ_metadata.csv")
    assert len(eq) == 8
    with h5py.File(tmp_path / "out/STEAD/STEAD_dataset_90s_singleEQ.h5") as hf:
        r = eq.iloc[0]
        assert hf[f"singleEQ/{r.split}/waveforms"][int(r.hdf5_index)].shape == (3, 9000)


def _fake_ceed(root: Path):
    """Two SC year files: 7 train + 3 dev traces, then 6 test + 2 dev traces."""
    rng = np.random.default_rng(3)
    for year, splits in (
        ("2019_0", ["train"] * 7 + ["dev"] * 3),
        ("2021", ["test"] * 6 + ["dev"] * 2),
    ):
        rows = []
        with h5py.File(root / f"waveformssc{year}.hdf5", "w") as hf:
            for k, sp in enumerate(splits):
                ev, st = f"ci{year}{k:03d}", f"CI.S{k:02d}..HH"
                hf.create_group(ev).create_dataset(
                    st, data=rng.normal(0, 1, (3, 12000)).astype(np.float32)
                )
                p = int(rng.integers(1500, 4000))
                s = p + int(rng.integers(300, 4000))
                rows.append(
                    dict(
                        trace_name=f"{ev}/{st}",
                        trace_p_arrival_sample=p,
                        trace_s_arrival_sample=s,
                        split=sp,
                        trace_phase_type_list="['P' 'S']",
                        trace_phase_arrival_sample_list=f"[{p} {s}]",
                    )
                )
        pd.DataFrame(rows).to_csv(root / f"metadatasc{year}.csv", index=False)


def test_ceed_convert(tmp_path):
    assert "usage" in run("-m", "builder.ceed_convert", "--help").lower()
    cache = tmp_path / "ceed"
    cache.mkdir()
    _fake_ceed(cache)
    env = {
        **ENV,
        "RPM_BENCH_CEED_CACHE": str(cache),
        "RPM_BENCH_CEED_SC_H5_DIR": str(tmp_path / "sc"),
    }
    run("-m", "builder.ceed_convert", "--region", "sc", "--shard-size", "3", "--seed", "1", env=env)
    shards = sorted((tmp_path / "sc").glob("CEED_SC_singleEQ_s*_metadata.csv"))
    meta = pd.concat([pd.read_csv(c) for c in shards])
    assert len(meta) == 18 and meta.split.value_counts().to_dict() == {
        "train": 7,
        "val": 5,
        "test": 6,
    }
    with h5py.File(shards[0].with_name(shards[0].name.replace("_metadata.csv", ".h5"))) as hf:
        sp = list(hf["singleEQ"])[0]
        assert hf[f"singleEQ/{sp}/waveforms"].shape[1:] == (3, 9000)
    # the original numbering overwrites shards across splits
    legacy = tmp_path / "legacy"
    run("-m", "builder.ceed_convert", "--region", "sc", "--shard-size", "3", "--seed", "1",
        "--legacy-shard-index", "--out-dir", str(legacy), env=env)  # fmt: skip
    kept = sum(len(pd.read_csv(c)) for c in legacy.glob("*_metadata.csv"))
    assert kept < 18
