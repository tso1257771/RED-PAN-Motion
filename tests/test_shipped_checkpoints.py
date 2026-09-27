"""The released checkpoints ship inside the package and load by name.

A plain ``pip install`` of the package, from a clone or from GitHub, must be
enough to run any of the three models: no separate download and no path into a
repository checkout.
"""
from pathlib import Path

import pytest

from redpan_motion.checkpoints import (
    CHECKPOINT_DIR,
    NAMES,
    checkpoint_dir,
    checkpoint_path,
    resolve,
)
from redpan_motion.inference import REDPANPredictor

MODEL_CLASS = {
    "redpan_60s": "Redpan60s",
    "redpan_motion": "MTAN_R2UNet_RP90_Motion",
    "edge_rp90": "EdgeRP90",
}


def test_shipped_directory_is_inside_the_package():
    import redpan_motion
    assert CHECKPOINT_DIR.parent == Path(redpan_motion.__file__).resolve().parent


@pytest.mark.parametrize("name", NAMES)
def test_each_checkpoint_has_weights_and_config(name):
    assert checkpoint_path(name).is_file()
    assert (checkpoint_dir(name) / "config.json").is_file()


@pytest.mark.parametrize("name", NAMES)
def test_predictor_loads_by_name(name):
    p = REDPANPredictor.from_checkpoint(name, device="cpu")
    assert type(p.model).__name__ == MODEL_CLASS[name]


def test_a_path_still_loads():
    p = REDPANPredictor.from_checkpoint(str(checkpoint_path("edge_rp90")), device="cpu")
    assert type(p.model).__name__ == "EdgeRP90"


def test_a_bare_name_is_not_shadowed_by_a_local_directory(tmp_path, monkeypatch):
    # Inside a RED-PAN-Motion checkout, ./redpan_motion is the package itself.
    (tmp_path / "redpan_motion").mkdir()
    monkeypatch.chdir(tmp_path)
    assert resolve("redpan_motion", file=False) == checkpoint_dir("redpan_motion")
    p = REDPANPredictor.from_checkpoint("redpan_motion", device="cpu")
    assert type(p.model).__name__ == "MTAN_R2UNet_RP90_Motion"


def test_other_values_are_paths():
    assert resolve("some/dir", file=False) == Path("some/dir")
    assert resolve(Path("redpan_motion"), file=False) == Path("redpan_motion")


def test_an_unknown_name_says_what_exists():
    with pytest.raises(KeyError, match="redpan_motion"):
        checkpoint_dir("redpan_90s")


@pytest.mark.parametrize("name", NAMES)
def test_seisbench_wrappers_load_by_name(name):
    sb = pytest.importorskip("redpan_motion.integrations.seisbench")
    cls = sb.RedpanSB60s if name == "redpan_60s" else sb.RedpanSB90s
    model = cls.from_redpan_checkpoint(name)
    assert model.variant == name
