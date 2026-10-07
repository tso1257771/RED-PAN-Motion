"""Data and output locations for the benchmark harness.

Every location comes from one place, resolved in this order (later wins):

1. the defaults below, relative to ``data_root`` (default ``../data``, the repository's ignored
   ``data/`` when run from ``benchmarks/``) and ``results_root`` (default ``./results``);
2. a YAML file: ``--config PATH`` on any entry point, or the ``RPM_BENCH_CONFIG`` environment
   variable (see ``benchmarks/config.example.yaml``);
3. environment variables ``RPM_BENCH_<KEY>`` (for example ``RPM_BENCH_H5_ROOT``).

Values may reference other keys as ``{data_root}``, ``{h5_root}`` and so on. Relative paths are
kept relative, so they resolve against the current directory: run the entry points from
``benchmarks/`` or give absolute paths.
"""

from __future__ import annotations

import argparse
import logging
import os
import string
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULTS: dict[str, str] = {
    # roots
    "data_root": "../data",
    "results_root": "./results",
    # 90 s HDF5 archives: <h5_root>/<DATASET>/<DATASET>_dataset_90s_{singleEQ,noise}.h5 (+ _metadata.csv)
    "h5_root": "{data_root}/input_h5_90sec_v3",
    # official STEAD test noise as 90 s windows (build_stead_noise_test.py)
    "stead_noise_test_dir": "{h5_root}/STEAD/STEAD_noise_test_90s",
    # original STEAD release: merge.hdf5, merge.csv, test.npy
    "stead_root": "{data_root}/STEAD",
    # GeoNet 2013-2014 holdout: metadata.csv, metadata_noise.csv, waveforms/, noise_waveforms/
    "geonet_holdout_root": "{data_root}/GeoNet_benchmark_test",
    # GeoNet benchmark dataset the holdout is built from (build_geonet_holdout.py)
    "geonet_source_root": "{data_root}/GeoNet_benchmark",
    # SeisBench CEED cache (metadata*.csv, waveforms*.hdf5) and the 99,998-trace test list
    "ceed_cache": "{data_root}/seisbench_cache/datasets/ceed",
    "ceed_test_ids": "{data_root}/ceed_test_ids_100k.npy",
    # streaming replay inputs: EQ_Mw4_to_6.9/ and noise/sac_120s_{CWB_StrongMotion,Palert}/
    "eew_root": "{data_root}/TW_EEW",
    # the RED-PAN 2022 paper model converted to PyTorch (README, "RED-PAN 2022 model")
    "redpan_paper_ckpt": "{data_root}/checkpoints/redpan_60s_paper/best.pt",
}


def _read_yaml(path: Path) -> dict[str, str]:
    """The ``paths:`` mapping of a YAML config (or its top level), values as strings."""
    import yaml  # PyYAML

    with open(path) as fh:
        d = yaml.safe_load(fh) or {}
    if not isinstance(d, dict):
        raise SystemExit(f"config file {path}: expected a mapping of keys to paths")
    return {str(k): str(v) for k, v in (d.get("paths", d) or {}).items() if v is not None}


def _references(key: str, value: str) -> list[str]:
    """Names of the ``{key}`` references in one value."""
    try:
        fields = [f for _, f, _, _ in string.Formatter().parse(value) if f is not None]
    except ValueError as e:  # unbalanced braces
        raise ValueError(f"config key {key!r}: cannot parse {value!r}: {e}") from None
    return [f.split(".")[0].split("[")[0] for f in fields]


def load(config_path: str | None = None) -> dict[str, Path]:
    """Resolve every location to a ``Path`` (defaults, then the YAML file, then ``RPM_BENCH_<KEY>``).

    Unknown YAML keys are kept (they may be referenced as ``{key}``) but logged as a warning.
    Raises ``SystemExit`` if the config file does not exist and ``ValueError`` if a ``{key}``
    reference names an unknown key or is still unresolved after five passes (a cycle).
    """
    raw = dict(DEFAULTS)
    path = config_path or os.environ.get("RPM_BENCH_CONFIG")
    if path:
        if not Path(path).is_file():
            src = "--config" if config_path else "RPM_BENCH_CONFIG"
            raise SystemExit(f"config file not found: {path} (from {src})")
        from_yaml = _read_yaml(Path(path))
        unknown = sorted(set(from_yaml) - set(DEFAULTS))
        if unknown:
            log.warning(
                "config %s: unknown key(s) %s (known: %s); they are only used where "
                "another value references them as {key}",
                path,
                ", ".join(unknown),
                ", ".join(DEFAULTS),
            )
        raw.update(from_yaml)
    for k in list(raw):
        env = os.environ.get(f"RPM_BENCH_{k.upper()}")
        if env:
            raw[k] = env
    for k, v in raw.items():
        for name in _references(k, v):
            if name not in raw:
                raise ValueError(
                    f"config key {k!r} = {v!r} references unknown key {{{name}}} "
                    f"(known keys: {', '.join(sorted(raw))})"
                )
    for _ in range(5):  # resolve {key} references
        raw = {k: v.format(**raw) for k, v in raw.items()}
    left = {k: v for k, v in raw.items() if _references(k, v)}
    if left:
        raise ValueError(f"unresolved {{key}} references after 5 passes (circular?): {left}")
    return {k: Path(os.path.expanduser(v)) for k, v in raw.items()}


def add_config_arg(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument(
        "--config",
        default=None,
        help="YAML file with data/output locations (default: $RPM_BENCH_CONFIG, else the "
        "built-in defaults ../data and ./results, relative to the current "
        "directory; RPM_BENCH_<KEY> overrides)",
    )
    return p


def results_dir(cfg: dict[str, Path], *parts: str) -> Path:
    """``<results_root>/<parts...>``, created on demand."""
    d = cfg["results_root"].joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def no_records_error(path: Path, key: str) -> SystemExit:
    """The error a runner raises when it read nothing: names the location and its config key."""
    return SystemExit(
        f"no records read from {path}; check config key {key} / RPM_BENCH_{key.upper()}"
    )
