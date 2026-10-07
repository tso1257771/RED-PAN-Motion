"""Paths of the run outputs the scorers read (all under ``results_root``):

static/<model>/<dataset>.csv                  run_static.py
noise/<model>/noisefp[_triggers]_<POOL>.csv   run_noise.py
native/<model>/<dataset>.csv                  run_native.py
polarity/<model>_polarity_per_event.csv       run_polarity.py
streaming/<model>/{eq,noise}/*.csv            run_streaming.py
"""

from __future__ import annotations

import json
from pathlib import Path

POOLS = ("STEAD_test", "GeoNet_test", "INSTANCE_test", "RockNet_test", "TW_test")  # 92,213 records
# Table II; GeoNet = the 2013-2014 holdout (70,583 earthquakes), as in the manuscript
TABLE2_SETS = ("ceed_nc", "ceed_sc", "crew", "geonet", "instance", "romplus", "stead_h5", "tw")
HELDOUT_SETS = (
    "ceed_nc",
    "ceed_sc",
    "crew",
    "geonet",
    "instance",
    "romplus",
    "stead_h5",
    "tw",
)  # Table IV, II-E
SHARED_SETS = ("stead", "geonet", "crew", "tw", "instance", "romplus")  # Table III, Fig. 3
NAME = {
    "ceed_nc": "CEED-NC",
    "ceed_sc": "CEED-SC",
    "crew": "CREW",
    "geonet": "GeoNet",
    "geonet_val": "GeoNet",
    "instance": "INSTANCE",
    "romplus": "ROMPLUS",
    "stead": "STEAD",
    "stead_h5": "STEAD",
    "tw": "TW",
}


def static_csv(cfg: dict, model: str, dataset: str) -> Path:
    return cfg["results_root"] / "static" / model / f"{dataset}.csv"


def noise_csv(cfg: dict, model: str, pool: str, triggers: bool = True) -> Path:
    return (
        cfg["results_root"]
        / "noise"
        / model
        / f"noisefp_{'triggers_' if triggers else ''}{pool}.csv"
    )


def native_csv(cfg: dict, model: str, dataset: str) -> Path:
    return cfg["results_root"] / "native" / model / f"{dataset}.csv"


def polarity_csv(cfg: dict, model: str) -> Path:
    return cfg["results_root"] / "polarity" / f"{model}_polarity_per_event.csv"


def stead_heldout(cfg: dict):
    """(clean_eq, test_noise): STEAD test.npy earthquakes that are not in the 90 s training or
    validation split (16,301), and the test.npy noise names (23,526)."""
    import numpy as np
    import pandas as pd

    md = pd.read_csv(
        cfg["h5_root"] / "STEAD" / "STEAD_dataset_90s_singleEQ_metadata.csv",
        usecols=["sample_id", "split"],
        low_memory=False,
    )
    core = lambda s: (str(s).removeprefix("singleEQ_")).removesuffix("_EV")
    seen = set(md.assign(c=md.sample_id.map(core)).query("split in ['train','val']").c)
    # test.npy of the STEAD release is an object array of trace names, so it needs
    # allow_pickle; check its md5 (README, Data) before loading a copy from elsewhere.
    t = np.load(cfg["stead_root"] / "test.npy", allow_pickle=True)
    clean = {str(x) for x in t if str(x).endswith("_EV") and str(x)[:-3] not in seen}
    noise = {str(x) for x in t if not str(x).endswith("_EV")}
    return clean, noise


def write_json(path: Path, obj) -> None:
    """Write a scorer's results as indented JSON, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)
