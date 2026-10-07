"""Deterministic train/val/test split helpers for builder adapters.

The legacy adapters used a per-row Bernoulli split (`rng.random() < train_frac`),
which has two problems:

  1. No test split — only train + val.
  2. Two samples from the same source event can land in different splits,
     causing event-level leakage.

This module provides three drop-in helpers:

  - `bernoulli_3way(seed, p_train, p_val)` — per-row 3-way split, fast and
    no-key-required. OK for noise where samples are independent.
  - `hash_split(key, p_train, p_val, salt)` — group samples that share `key`
    (e.g. event_id, source_id) into the SAME split. Stable across runs given
    the same salt.
  - `year_split(year, train_years, val_years, test_years)` — explicit year
    assignment for catalogues with year-based partitioning (GeoNet style).

Default proportions: 70 % train / 15 % val / 15 % test.

Usage in an adapter:

    from builder.splits import hash_split

    for row in metadata.iterrows():
        split = hash_split(row['source_id'], salt='STEAD-singleEQ-v1')
        yield WaveformSample(..., split=split)
"""
from __future__ import annotations
import hashlib
from typing import Iterable, Literal, Optional

import numpy as np


Split = Literal["train", "val", "test"]

_DEFAULT_TRAIN = 0.70
_DEFAULT_VAL   = 0.15
# test = 1 - train - val


def bernoulli_3way(
    rng: np.random.Generator,
    p_train: float = _DEFAULT_TRAIN,
    p_val: float = _DEFAULT_VAL,
) -> Split:
    """Single-sample Bernoulli draw. Use only when samples are independent
    (e.g. noise traces with no shared event source)."""
    u = float(rng.random())
    if u < p_train:
        return "train"
    if u < p_train + p_val:
        return "val"
    return "test"


def hash_split(
    key: str,
    p_train: float = _DEFAULT_TRAIN,
    p_val: float = _DEFAULT_VAL,
    salt: str = "redpan-builder",
) -> Split:
    """Deterministic split: every sample with the same `key` lands in the
    same split. Use for event-bearing categories where multiple traces share
    a source event.

    The hash is BLAKE2s of `salt|key` interpreted as a uint64; modulo a high
    denominator gives uniform distribution.
    """
    h = hashlib.blake2s(f"{salt}|{key}".encode("utf-8"), digest_size=8).digest()
    u = int.from_bytes(h, "big") / 2**64       # uniform on [0, 1)
    if u < p_train:
        return "train"
    if u < p_train + p_val:
        return "val"
    return "test"


def year_split(
    year: int,
    train_years: Iterable[int],
    val_years: Iterable[int],
    test_years: Iterable[int],
) -> Optional[Split]:
    """Return the split for a given year, or None if year is not listed.

    Suggested for GeoNet:
        train_years = range(2015, 2022)
        val_years   = (2022, 2023)
        test_years  = (2024,)
    """
    if year in set(train_years):
        return "train"
    if year in set(val_years):
        return "val"
    if year in set(test_years):
        return "test"
    return None


def assert_valid_proportions(p_train: float, p_val: float) -> None:
    """Sanity-check split proportions."""
    if not 0.0 < p_train < 1.0:
        raise ValueError(f"p_train must be in (0,1); got {p_train}")
    if not 0.0 <= p_val < 1.0 - p_train:
        raise ValueError(
            f"p_val must be in [0, {1.0 - p_train}); got {p_val}"
        )
