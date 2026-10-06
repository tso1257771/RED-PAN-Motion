"""The released checkpoints, shipped inside the package.

Each ``<name>/`` directory holds ``best.pt`` and the ``config.json`` that
rebuilds its architecture:

    redpan_60s     RED-PAN 60 s, release REDPAN_60s_240107, ported to PyTorch
    redpan_motion  the 90 s model with first-motion polarity
    edge_rp90      the smaller 90 s model for edge devices

``REDPANPredictor.from_checkpoint`` and the SeisBench wrappers'
``from_redpan_checkpoint`` accept these names in place of a path. A bare name
always means the shipped checkpoint; to load a directory of the same name, pass
a path such as ``./redpan_motion``.
"""
from __future__ import annotations

from pathlib import Path

CHECKPOINT_DIR = Path(__file__).resolve().parent
NAMES = ("redpan_60s", "redpan_motion", "edge_rp90")


def checkpoint_dir(name: str) -> Path:
    """The directory of a shipped checkpoint."""
    if name not in NAMES:
        raise KeyError(f"no shipped checkpoint {name!r}; available: {', '.join(NAMES)}")
    return CHECKPOINT_DIR / name


def checkpoint_path(name: str) -> Path:
    """The ``best.pt`` of a shipped checkpoint."""
    return checkpoint_dir(name) / "best.pt"


def resolve(path_or_name: str | Path, *, file: bool) -> Path:
    """A shipped checkpoint name becomes its ``best.pt`` (``file=True``) or its
    directory; any other value is a path and is returned unchanged."""
    if isinstance(path_or_name, str) and path_or_name in NAMES:
        return checkpoint_path(path_or_name) if file else checkpoint_dir(path_or_name)
    return Path(path_or_name)


__all__ = ["CHECKPOINT_DIR", "NAMES", "checkpoint_dir", "checkpoint_path", "resolve"]
