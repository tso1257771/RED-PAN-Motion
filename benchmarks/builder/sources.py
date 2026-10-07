"""Source locations and the build plan for the 90 s archives.

Locations resolve exactly like ``rpm_bench.config`` (defaults < YAML file from ``--config`` or
``RPM_BENCH_CONFIG`` < environment ``RPM_BENCH_<KEY>``), with the source keys below added to the
harness keys (``data_root``, ``h5_root``, ``stead_root``, ``geonet_source_root``, ...).

``PLAN`` lists every (dataset, category) archive the 90 s models were trained and tested on, with
the source files each adapter reads and the options of the original regeneration run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional

from rpm_bench import config as harness_config

SOURCE_DEFAULTS: Dict[str, str] = {
    # rebuilt archives: <build_root>/<DATASET>/<DATASET>_dataset_90s_<CATEGORY>.h5 (+ _metadata.csv).
    # Kept apart from h5_root so a rebuild never overwrites the archive the benchmark reads.
    "build_root": "{data_root}/input_h5_90sec_rebuilt",
    # INSTANCE release: Instance_events_gm.hdf5, metadata_Instance_events_v2.csv,
    # Instance_noise.hdf5, metadata_Instance_noise.csv
    "instance_root": "{data_root}/INSTANCE",
    # SeisBench CREW download: chunks, metadata{NNN}.csv, waveforms{NNN}.hdf5
    "crew_root": "{data_root}/seisbench_cache/datasets/crew",
    # OBSTransformer training_data.hdf5
    "obst_root": "{data_root}/OBSTransformer",
    # RockNet data/: Luhu_hdf5/Luhu_dataset.h5 and metadata/partition/*_partition.npy
    "rocknet_root": "{data_root}/RockNet",
    # ROMPLUS: sac/{year}/{event}/*.sac and romplus_metadata.csv
    "romplus_root": "{data_root}/ROMPLUS",
    # CEED as 90 s windows (intermediate H5s the CEED adapters repack; see README)
    "ceed_nc_h5_dir": "{data_root}/CEED_NC_redpan_h5/CEED_NC",
    "ceed_sc_h5_dir": "{data_root}/CEED_SC_redpan_h5/CEED_SC",
    # Taiwan (restricted): TW_2012_2019_180s/, metadata_TW_2012_2019_180s/, first_motion/available_data.csv
    "tw_eq_root": "{data_root}/TW_eq_data",
    # Taiwan noise (restricted): metadata/TW_noise/ (hourly SAC) and metadata/pred_TW_noise/
    "tw_noise_root": "{data_root}/TW_noise_data",
}


def load(config_path: Optional[str] = None) -> Dict[str, Path]:
    """Harness keys plus the source keys, resolved by ``rpm_bench.config.load`` (same precedence,
    same checks): its ``DEFAULTS`` are extended with ``SOURCE_DEFAULTS`` for the call."""
    saved = harness_config.DEFAULTS
    harness_config.DEFAULTS = {**SOURCE_DEFAULTS, **saved}
    try:
        return harness_config.load(config_path)
    finally:
        harness_config.DEFAULTS = saved


class Sources(NamedTuple):
    source: Path
    metadata: Optional[Path] = None
    extra_source: Optional[Path] = None
    extra_metadata: Optional[Path] = None
    options: Dict[str, object] = {}


def _stead(c, cat):
    s = c["stead_root"]
    if cat == "noise":
        # the official test noise (test.npy) is excluded and the rest split 80/20 train/val,
        # the membership of the archive's STEAD noise (211,900 = 235,426 - 23,526 records)
        return Sources(
            s / "merge.hdf5",
            s / "merge.csv",
            options=dict(exclude_test_npy=s / "test.npy", p_train=0.8, p_val=0.2),
        )
    return Sources(s / "merge.hdf5", s / "merge.csv")


def _instance(c, cat):
    r = c["instance_root"]
    return Sources(
        r / "Instance_events_gm.hdf5",
        r / "metadata_Instance_events_v2.csv",
        r / "Instance_noise.hdf5",
        r / "metadata_Instance_noise.csv",
    )


def _geonet(c, cat):
    r = c["geonet_source_root"]
    if cat == "noise":
        return Sources(
            r / "noise_dataset/waveform_data/units/waveforms_units_noise.h5",
            r / "noise_dataset/metadata/metadata_noise.csv",
        )
    return Sources(r / "event_dataset/waveform_data/units", r / "event_dataset/metadata")


def _crew(c, cat):
    return Sources(
        c["crew_root"], options=dict(crew_phase_pair="mantle") if cat == "singleEQ" else {}
    )


def _tw(c, cat):
    if cat == "noise":
        r = c["tw_noise_root"] / "metadata"
        return Sources(r / "TW_noise", r / "pred_TW_noise")
    r = c["tw_eq_root"]
    pol = r / "first_motion/available_data.csv" if cat in ("singleEQ", "Ponly") else None
    return Sources(r / "TW_2012_2019_180s", r / "metadata_TW_2012_2019_180s", extra_metadata=pol)


PLAN: Dict[tuple, Callable] = {
    ("STEAD", "noise"): _stead,
    ("STEAD", "singleEQ"): _stead,
    ("INSTANCE", "noise"): _instance,
    ("INSTANCE", "singleEQ"): _instance,
    ("GeoNet", "noise"): _geonet,
    ("GeoNet", "singleEQ"): _geonet,
    ("CREW", "singleEQ"): _crew,
    ("CREW", "Ponly"): _crew,
    ("CREW", "Sonly"): _crew,
    ("OBSTransformer", "singleEQ"): lambda c, cat: Sources(c["obst_root"] / "training_data.hdf5"),
    ("RockNet", "noise"): lambda c, cat: Sources(
        c["rocknet_root"] / "Luhu_hdf5/Luhu_dataset.h5", c["rocknet_root"] / "metadata/partition"
    ),
    ("ROMPLUS", "singleEQ"): lambda c, cat: Sources(
        c["romplus_root"] / "sac", c["romplus_root"] / "romplus_metadata.csv"
    ),
    ("ROMPLUS", "Ponly"): lambda c, cat: Sources(
        c["romplus_root"] / "sac", c["romplus_root"] / "romplus_metadata.csv"
    ),
    ("ROMPLUS", "Sonly"): lambda c, cat: Sources(
        c["romplus_root"] / "sac", c["romplus_root"] / "romplus_metadata.csv"
    ),
    ("CEED_NC", "singleEQ"): lambda c, cat: Sources(
        c["ceed_nc_h5_dir"] / "CEED_NC_dataset_90s_singleEQ.h5",
        c["ceed_nc_h5_dir"] / "CEED_NC_dataset_90s_singleEQ_metadata.csv",
        options=dict(with_polarity=True),
    ),
    ("CEED_SC", "singleEQ"): lambda c, cat: Sources(
        c["ceed_sc_h5_dir"], options=dict(with_polarity=True)
    ),
    ("TW", "noise"): _tw,
    ("TW", "singleEQ"): _tw,
    ("TW", "Ponly"): _tw,
    ("TW", "Sonly"): _tw,
}


def plan_sources(cfg: Dict[str, Path], dataset: str, category: str) -> Sources:
    return PLAN[(dataset, category)](cfg, category)


def output_paths(cfg: Dict[str, Path], dataset: str, category: str):
    d = cfg["build_root"] / dataset
    stem = f"{dataset}_dataset_90s_{category}"
    return d / f"{stem}.h5", d / f"{stem}_metadata.csv"
