"""Builder for the 90 s HDF5 archives the benchmark reads (``<h5_root>/<DATASET>/``).

Ported from RED-PAN's ``redpan/data/builder``: one schema (``WaveformSample``), one writer
(``build_h5``) and one adapter per (dataset, category). Adapters read the original dataset
releases, window them to 90 s, pad with spectrum-matched noise and assign the splits.
Command line: ``python -m builder --help`` (from ``benchmarks/``); see ``builder/README.md``.
"""

from .adapters.base import BaseAdapter
from .schema import WaveformSample
from .splits import bernoulli_3way, hash_split, year_split
from .validators import assert_h5_consistency
from .writer import build_h5

__all__ = [
    "WaveformSample",
    "BaseAdapter",
    "build_h5",
    "assert_h5_consistency",
    "bernoulli_3way",
    "hash_split",
    "year_split",
]
