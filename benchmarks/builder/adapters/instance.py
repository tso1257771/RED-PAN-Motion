"""INSTANCE noise + singleEQ adapters reading directly from Instance_*.hdf5.

INSTANCE traces are 120 s (12000 samples) so we randomly slice 90 s windows —
no synthetic padding needed for either category.

Splitting policy (matches stead.py)
-----------------------------------
* Noise rows are independent → per-row Bernoulli draw via
  ``bernoulli_3way(rng, p_train, p_val)``.
* Single-event rows share a ``source_id`` (one earthquake observed at many
  stations).  Using ``hash_split`` keyed on ``source_id`` keeps every
  station-recording of the same event in the same split, eliminating the
  same-event leakage that the per-row Bernoulli split previously caused.

Bugfixes from the 2026-04 audit
-------------------------------
INSTANCE-3
    Skip rows whose ``trace_dt_s`` is not 0.01 s (i.e. not 100 Hz).  The
    current INSTANCE shipment is uniformly 100 Hz, but the guard is cheap
    and survives future re-issues.
INSTANCE-4
    Pick indices in the metadata CSV are stored as floats.  Truncating with
    ``int(...)`` introduces a systematic ~0.5-sample bias toward earlier
    samples; switch to ``int(round(float(...)))``.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional
import logging

import h5py
import numpy as np
import pandas as pd

from .base import BaseAdapter
from ..schema import (
    WaveformSample, T_SAMPLES, N_CHANNELS, SAMPLING_RATE_HZ,
)
from ..splits import bernoulli_3way, hash_split


log = logging.getLogger(__name__)


# 100 Hz tolerance — INSTANCE catalogue lists trace_dt_s as float.
_DT_S_NOMINAL = 1.0 / SAMPLING_RATE_HZ            # 0.01
_DT_S_ATOL = 1e-6                                 # |dt - 0.01| < 1e-6 ⇒ accept


def _ps_diff_bin(ps_diff_samples: int) -> str:
    """Convert P-S sample difference into a sampling_category bin name
    matching the train_*.json category_weights keys."""
    sec = ps_diff_samples / SAMPLING_RATE_HZ
    if sec < 5:   return "singleEQ_00-05s"
    if sec < 10:  return "singleEQ_05-10s"
    if sec < 15:  return "singleEQ_10-15s"
    if sec < 20:  return "singleEQ_15-20s"
    return "singleEQ_20s_plus"


def _polarity_from_str(s: str) -> str:
    """Map INSTANCE's trace_polarity into our {'', 'U', 'D'} dictionary."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    if s in ("up", "positive", "u", "+"):
        return "U"
    if s in ("down", "negative", "d", "-"):
        return "D"
    return ""   # 'undecidable' / 'unknown' / NaN


def _is_100hz(row: pd.Series) -> bool:
    """Audit guard INSTANCE-3: only accept 100 Hz traces.

    Reads the metadata CSV's ``trace_dt_s`` column; missing column is treated
    as a fail-closed reject."""
    dt = row.get("trace_dt_s", np.nan)
    try:
        dt_f = float(dt)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(dt_f):
        return False
    return abs(dt_f - _DT_S_NOMINAL) < _DT_S_ATOL


class INSTANCENoiseAdapter(BaseAdapter):
    """INSTANCE noise: slice random 9000-sample window from 12000-sample trace.

    Noise rows are independent (no shared source event), so we use a per-row
    Bernoulli 3-way split via :func:`bernoulli_3way`.
    """
    name = "INSTANCE"
    category = "noise"

    def __init__(
        self,
        events_hdf5: Path,
        noise_hdf5: Path,
        events_csv: Path,
        noise_csv: Path,
        *,
        p_train: float = 0.70,
        p_val: float = 0.15,
        seed: int = 42,
        **kwargs,
    ):
        """``p_test = 1 - p_train - p_val`` (default 0.15)."""
        super().__init__(**kwargs)
        self.noise_hdf5 = Path(noise_hdf5)
        self.noise_csv = Path(noise_csv)
        # events_hdf5/csv unused here but accepted for symmetric signature
        self.p_train = p_train
        self.p_val = p_val
        self.seed = seed

    def __iter__(self) -> Iterator[WaveformSample]:
        log.info("INSTANCENoiseAdapter: reading %s", self.noise_csv)
        df = pd.read_csv(self.noise_csv, low_memory=False)
        log.info("  noise rows in metadata: %d", len(df))

        rng = np.random.default_rng(self.seed)

        n_yielded = 0
        n_dropped_dt = 0
        with h5py.File(self.noise_hdf5, "r") as hf:
            for _, row in df.iterrows():
                if self.max_samples is not None and n_yielded >= self.max_samples:
                    break

                # Audit INSTANCE-3 — sampling-rate guard.
                if not _is_100hz(row):
                    n_dropped_dt += 1
                    continue

                trace_name = str(row["trace_name"])
                split = bernoulli_3way(rng, self.p_train, self.p_val)
                if self.split_filter and split != self.split_filter:
                    continue
                try:
                    arr = np.array(hf["data"][trace_name]).astype(np.float32)
                except (KeyError, OSError):
                    continue

                # INSTANCE stores (3, 12000) channel-first (E, N, Z)
                if arr.shape != (3, 12000):
                    if arr.shape == (12000, 3):
                        arr = arr.T
                    else:
                        continue

                # Random 9000-sample slice from 12000 — no padding needed
                max_start = arr.shape[1] - T_SAMPLES
                start = int(rng.integers(0, max_start + 1))
                wf_cf = arr[:, start:start + T_SAMPLES].copy()
                if not np.isfinite(wf_cf).all():
                    continue

                yield WaveformSample(
                    sample_id=f"noise_{trace_name}_{start}",
                    waveform=wf_cf,
                    category="noise",
                    sampling_category="noise",
                    split=split,
                    source_file=str(self.noise_hdf5),
                    p_arrival_samples=[],
                    s_arrival_samples=[],
                    polarity="",
                )
                n_yielded += 1

        if n_dropped_dt:
            log.info(
                "INSTANCENoiseAdapter: dropped %d rows with trace_dt_s != 0.01 s",
                n_dropped_dt,
            )


class INSTANCESingleEQAdapter(BaseAdapter):
    """INSTANCE single-event: position P at random offset within 9000 window
    via slicing from the 12000-sample real trace.  No synthetic pad needed
    when the slice fits entirely inside the original trace.

    Single-event rows are split with ``hash_split`` keyed on ``source_id`` so
    every station recording of the same earthquake lands in the same split.
    """
    name = "INSTANCE"
    category = "singleEQ"

    def __init__(
        self,
        events_hdf5: Path,
        noise_hdf5: Path,
        events_csv: Path,
        noise_csv: Path,
        *,
        p_train: float = 0.70,
        p_val: float = 0.15,
        seed: int = 42,
        min_p_position: int = 500,
        max_p_position: int = 7500,
        split_salt: str = "INSTANCE-singleEQ-v1",
        **kwargs,
    ):
        """``p_test = 1 - p_train - p_val`` (default 0.15)."""
        super().__init__(**kwargs)
        self.events_hdf5 = Path(events_hdf5)
        self.events_csv = Path(events_csv)
        self.p_train = p_train
        self.p_val = p_val
        self.seed = seed
        self.min_p_position = min_p_position
        self.max_p_position = max_p_position
        self.split_salt = split_salt

    def __iter__(self) -> Iterator[WaveformSample]:
        log.info("INSTANCESingleEQAdapter: reading %s", self.events_csv)
        df = pd.read_csv(self.events_csv, low_memory=False)
        log.info("  EQ rows in metadata: %d", len(df))

        rng = np.random.default_rng(self.seed)

        n_yielded = 0
        n_dropped_dt = 0
        with h5py.File(self.events_hdf5, "r") as hf:
            for _, row in df.iterrows():
                if self.max_samples is not None and n_yielded >= self.max_samples:
                    break

                # Audit INSTANCE-3 — sampling-rate guard.
                if not _is_100hz(row):
                    n_dropped_dt += 1
                    continue

                trace_name = str(row["trace_name"])

                # Event-aware split: every trace from the same source_id lands
                # in the same split — prevents same-event leakage.  Falls back
                # to trace_name if source_id is absent.
                split_key = str(row.get("source_id", "") or trace_name)
                split = hash_split(
                    split_key, self.p_train, self.p_val, salt=self.split_salt,
                )
                if self.split_filter and split != self.split_filter:
                    continue

                # Audit INSTANCE-4 — round() not int() on fractional picks.
                try:
                    p_sample = int(round(float(row["trace_P_arrival_sample"])))
                    s_sample = int(round(float(row["trace_S_arrival_sample"])))
                except (TypeError, ValueError, KeyError):
                    continue
                if not (0 < p_sample < s_sample < 12000):
                    continue
                ps_residual = s_sample - p_sample

                try:
                    arr = np.array(hf["data"][trace_name]).astype(np.float32)
                except (KeyError, OSError):
                    continue
                if arr.shape != (3, 12000):
                    if arr.shape == (12000, 3):
                        arr = arr.T
                    else:
                        continue

                # Random target P position in output window
                target_p = int(rng.integers(self.min_p_position,
                                             self.max_p_position + 1))
                # We need slice_start such that p in slice = target_p,
                # i.e. slice_start = p_sample - target_p. Verify slice fits.
                slice_start = p_sample - target_p
                slice_end = slice_start + T_SAMPLES
                if slice_start < 0 or slice_end > arr.shape[1]:
                    continue
                wf_cf = arr[:, slice_start:slice_end].copy()
                if not np.isfinite(wf_cf).all():
                    continue

                new_p = target_p
                new_s = target_p + ps_residual
                if not (0 <= new_p < T_SAMPLES and 0 <= new_s < T_SAMPLES):
                    continue

                pol = _polarity_from_str(row.get("trace_polarity", ""))

                yield WaveformSample(
                    sample_id=f"singleEQ_{trace_name}_{slice_start}",
                    waveform=wf_cf,
                    category="singleEQ",
                    sampling_category=_ps_diff_bin(ps_residual),
                    split=split,
                    source_file=str(self.events_hdf5),
                    p_arrival_samples=[int(new_p)],
                    s_arrival_samples=[int(new_s)],
                    polarity=pol,
                )
                n_yielded += 1

        if n_dropped_dt:
            log.info(
                "INSTANCESingleEQAdapter: dropped %d rows with trace_dt_s != 0.01 s",
                n_dropped_dt,
            )
