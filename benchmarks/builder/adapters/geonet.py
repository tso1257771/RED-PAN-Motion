"""GeoNet noise + singleEQ adapters reading directly from per-year `waveforms_*.h5`.

GeoNet (New Zealand) source format
----------------------------------
- Event waveforms: ``event_dataset/waveform_data/units/waveforms_units_{year}.h5``
  (one HDF5 per year), 100 Hz, ``data/{trace_name}`` shape ``(3, 12000)``,
  channel-first, columns in ``station_component_order`` semantics → **(E, N, Z)**
  (some accelerometer channels use 1/2/Z naming but the component order metadata
  reports ENZ, matching our builder convention).
- Noise waveforms: ``noise_dataset/waveform_data/units/waveforms_units_noise.h5``,
  same ``(3, 12000)`` channel-first layout.
- Event metadata: per-year ``metadata_events_{year}.csv`` with columns
  ``trace_name``, ``trace_p_arrival_sample``, ``trace_s_arrival_sample``,
  ``trace_category``.  Many events have only P (S is NaN); those are dropped
  per the singleEQ contract (must have both picks).
- Noise metadata: ``metadata_noise.csv`` with ``trace_name``, ``trace_category``,
  ``trace_start_time``.

Splitting policy (replaces the legacy per-row Bernoulli draw)
-------------------------------------------------------------
Both adapters now use :func:`year_split` so that every recording captured in
year *Y* lands in the same split, regardless of which station observed it.

* ``train_years``  — default ``(2015, 2016, 2017, 2018, 2019, 2020, 2021)``
* ``val_years``    — default ``(2022, 2023)``
* ``test_years``   — default ``(2024,)``

Years 2013-2014 remain reserved for the external ``GeoNet_benchmark_test``
held-out set; rows from those years are skipped when their year is not listed
in any of the train/val/test tuples.

Bugfixes from the 2026-04 audit
-------------------------------
GeoNet-A
    NaN ``trace_s_arrival_sample`` rows are skipped explicitly with a single
    summary log line at iterator-start.  The singleEQ contract requires both
    picks.
GeoNet-B
    Rows where ``s_sample + min_back_buffer > T_SAMPLES`` (P-S residual too
    large to fit in the 9000-sample output window) are skipped explicitly
    with a debug log and a summary count, instead of silently failing later.
GeoNet-C
    The pick CSVs store ``trace_*_arrival_sample`` as floats; we now use
    ``int(round(float(...)))`` to avoid the systematic truncation bias.

GeoNet has no first-motion polarity labels and no Ponly/Sonly/EEWA/MMWA in
this dataset — those adapters are intentionally NOT provided.

This adapter mirrors the legacy GeoNet scripts ``P01_TFRecord_singleEQ.py`` and
``P05_TFRecord_noise.py`` (TFRecord generators, not distributed).
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Iterable, Optional, Sequence
import logging

import h5py
import numpy as np
import pandas as pd

from .base import BaseAdapter
from ..schema import (
    WaveformSample, T_SAMPLES, N_CHANNELS, SAMPLING_RATE_HZ,
)
from ..splits import year_split
from redpan_motion.utils.waveform import find_reference_signal, generate_matching_noise


log = logging.getLogger(__name__)


# Default year-based split.  2013-2014 are reserved for the external
# GeoNet_benchmark_test held-out set and are intentionally absent from
# all three of the train/val/test tuples.
DEFAULT_TRAIN_YEARS: tuple[int, ...] = (2015, 2016, 2017, 2018, 2019, 2020, 2021)
DEFAULT_VAL_YEARS:   tuple[int, ...] = (2022, 2023)
DEFAULT_TEST_YEARS:  tuple[int, ...] = (2024,)


def _ps_diff_bin(ps_diff_samples: int) -> str:
    """Convert P-S sample difference into a sampling_category bin name
    matching the train_*.json category_weights keys."""
    sec = ps_diff_samples / SAMPLING_RATE_HZ
    if sec < 5:   return "singleEQ_00-05s"
    if sec < 10:  return "singleEQ_05-10s"
    if sec < 15:  return "singleEQ_10-15s"
    if sec < 20:  return "singleEQ_15-20s"
    return "singleEQ_20s_plus"


def _coerce_3x12000(arr: np.ndarray, expected_npts: int) -> Optional[np.ndarray]:
    """Return a (3, expected_npts) float32 array, or None if the source can't
    be coerced. Handles the legacy (3, 12000) and the rare (12000, 3) cases."""
    if arr.ndim != 2:
        return None
    if arr.shape == (N_CHANNELS, expected_npts):
        out = arr
    elif arr.shape == (expected_npts, N_CHANNELS):
        out = arr.T
    else:
        return None
    return out.astype(np.float32, copy=False)


def _year_from_start_time(s: object) -> Optional[int]:
    """Extract the year from a GeoNet ``trace_start_time`` value.

    GeoNet stores this as ISO-8601 ``YYYY-MM-DDTHH:MM:SS...Z``.  Returns None
    when the value is missing or unparsable.
    """
    if not isinstance(s, str) or len(s) < 4:
        return None
    head = s[:4]
    try:
        return int(head)
    except ValueError:
        return None


class GeoNetNoiseAdapter(BaseAdapter):
    """GeoNet noise: 12000-sample real → random 9000-sample slice.

    GeoNet noise traces are already 120 s (12000 samples) at 100 Hz, so we
    simply slice a random 9000-sample (90 s) window — no synthetic padding
    needed.  Mirrors the legacy ``P05_TFRecord_noise.py`` SNR filter.

    Splitting is by year (extracted from ``trace_start_time``) via
    :func:`year_split`; rows whose year isn't listed in any of the train/val/
    test tuples are skipped (logged once at iterator-start).
    """
    name = "GeoNet"
    category = "noise"

    def __init__(
        self,
        noise_hdf5: Path,
        noise_csv: Path,
        *,
        original_npts: int = 12000,
        snr_threshold: float = 10.0,
        p_train: float = 0.70,
        p_val: float = 0.15,
        train_years: Iterable[int] = DEFAULT_TRAIN_YEARS,
        val_years:   Iterable[int] = DEFAULT_VAL_YEARS,
        test_years:  Iterable[int] = DEFAULT_TEST_YEARS,
        seed: int = 42,
        **kwargs,
    ):
        """``p_test = 1 - p_train - p_val`` (default 0.15).

        ``p_train`` / ``p_val`` are accepted for parity with the other
        adapters, but the actual GeoNet split is decided by year — set the
        ``*_years`` tuples to control which rows go where.
        """
        super().__init__(**kwargs)
        self.noise_hdf5 = Path(noise_hdf5)
        self.noise_csv = Path(noise_csv)
        self.original_npts = original_npts
        self.snr_threshold = snr_threshold
        self.p_train = p_train
        self.p_val = p_val
        self.train_years = tuple(train_years)
        self.val_years = tuple(val_years)
        self.test_years = tuple(test_years)
        self.seed = seed

    def __iter__(self) -> Iterator[WaveformSample]:
        log.info("GeoNetNoiseAdapter: reading %s", self.noise_csv)
        df = pd.read_csv(self.noise_csv, low_memory=False)
        df = df[df["trace_category"] == "noise"].copy()
        log.info("  noise rows in metadata: %d", len(df))

        # Compute year + split per row up-front so we can summarize how many
        # rows are dropped because their year is outside our train/val/test
        # tuples (held-out benchmark years).
        df["__year"] = df["trace_start_time"].map(_year_from_start_time)
        n_no_year = int(df["__year"].isna().sum())
        if n_no_year:
            log.info(
                "GeoNetNoiseAdapter: dropping %d rows with unparsable trace_start_time",
                n_no_year,
            )
        df = df.dropna(subset=["__year"]).copy()
        df["__year"] = df["__year"].astype(int)

        df["__split"] = df["__year"].map(
            lambda y: year_split(int(y), self.train_years, self.val_years, self.test_years)
        )
        n_held_out = int(df["__split"].isna().sum())
        if n_held_out:
            held_out_years = sorted(set(df.loc[df["__split"].isna(), "__year"]))
            log.info(
                "GeoNetNoiseAdapter: skipping %d rows from held-out years %s",
                n_held_out, held_out_years,
            )
        df = df.dropna(subset=["__split"]).copy()

        rng = np.random.default_rng(self.seed)

        n_yielded = 0
        with h5py.File(self.noise_hdf5, "r") as hf:
            data_grp = hf["data"]
            for _, row in df.iterrows():
                if self.max_samples is not None and n_yielded >= self.max_samples:
                    break
                trace_name = str(row["trace_name"])
                split = str(row["__split"])
                if self.split_filter and split != self.split_filter:
                    continue

                try:
                    arr = np.array(data_grp[trace_name])
                except (KeyError, OSError):
                    continue

                arr32 = _coerce_3x12000(arr, self.original_npts)
                if arr32 is None:
                    continue

                # SNR filter: skip traces whose Z-component shows a transient
                # — divide into 10 s windows and reject if max/min std ratio
                # exceeds threshold (matches P05 legacy logic).
                z = arr32[2]
                win = 1000  # 10 s @ 100 Hz
                n_wins = len(z) // win
                if n_wins >= 2:
                    win_stds = np.array(
                        [np.std(z[i * win:(i + 1) * win]) for i in range(n_wins)]
                    )
                    min_std = float(np.min(win_stds))
                    if min_std > 0:
                        snr_ratio = float(np.max(win_stds)) / min_std
                        if snr_ratio > self.snr_threshold:
                            continue

                # Random 9000-sample slice from 12000 — no padding needed.
                max_start = arr32.shape[1] - T_SAMPLES
                start = int(rng.integers(0, max_start + 1))
                wf_cf = arr32[:, start:start + T_SAMPLES].copy()
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


class GeoNetSingleEQAdapter(BaseAdapter):
    """GeoNet single-event: position P at random offset within 9000 window.

    GeoNet event traces are 12000 samples at 100 Hz.  We choose a random
    target P position in the output, then slice 9000 samples ending such
    that the original P lands exactly at that target.  If the slice would
    fall outside the original 12000-sample trace, we fall back to spectrum-
    matched padding on the deficient side(s) (legacy P01 behaviour).

    Splitting is by year (already known from the per-year metadata CSV) via
    :func:`year_split`; rows whose year isn't listed in any of the train/val/
    test tuples are skipped at metadata-load time.

    Notes
    -----
    - Drops events whose ``trace_s_arrival_sample`` is NaN with a single
      summary log line (audit GeoNet-A).
    - Drops events whose P-S residual would push S past the right edge of
      the output window with a debug log + summary count (audit GeoNet-B).
    - Pick samples are parsed via ``int(round(float(...)))`` (audit GeoNet-C).
    - GeoNet does not provide first-motion polarity, so ``polarity=''``.
    - Years 2013-2014 are reserved for the external benchmark test set; the
      default ``train_years`` / ``val_years`` / ``test_years`` tuples leave
      those years out so they cannot leak.
    """
    name = "GeoNet"
    category = "singleEQ"

    def __init__(
        self,
        events_hdf5_dir: Path,
        events_csv_dir: Path,
        *,
        years: Optional[Iterable[int]] = None,
        original_npts: int = 12000,
        p_train: float = 0.70,
        p_val: float = 0.15,
        train_years: Iterable[int] = DEFAULT_TRAIN_YEARS,
        val_years:   Iterable[int] = DEFAULT_VAL_YEARS,
        test_years:  Iterable[int] = DEFAULT_TEST_YEARS,
        seed: int = 42,
        min_p_position: int = 1000,
        max_p_position: int = 7000,
        min_back_buffer: int = 0,
        **kwargs,
    ):
        """``p_test = 1 - p_train - p_val`` (default 0.15).

        ``p_train`` / ``p_val`` are accepted for parity with other adapters,
        but the actual GeoNet split is decided by year.

        Parameters
        ----------
        years
            Backward-compat: explicit override of the years to ingest.  When
            ``None`` (default), ingests the union of ``train_years``,
            ``val_years``, and ``test_years``.
        min_back_buffer
            Minimum number of samples that must remain after the S pick in
            the output window.  A row with S past ``T_SAMPLES - min_back_buffer``
            is dropped early with a summary log (audit GeoNet-B).
        """
        super().__init__(**kwargs)
        self.events_hdf5_dir = Path(events_hdf5_dir)
        self.events_csv_dir = Path(events_csv_dir)
        self.train_years = tuple(train_years)
        self.val_years = tuple(val_years)
        self.test_years = tuple(test_years)
        self.years: tuple[int, ...] = tuple(
            years if years is not None
            else (*self.train_years, *self.val_years, *self.test_years)
        )
        self.original_npts = original_npts
        self.p_train = p_train
        self.p_val = p_val
        self.seed = seed
        self.min_p_position = min_p_position
        self.max_p_position = max_p_position
        self.min_back_buffer = int(min_back_buffer)

    def _load_metadata(self) -> pd.DataFrame:
        frames = []
        for year in self.years:
            csv_path = self.events_csv_dir / f"metadata_events_{year}.csv"
            if not csv_path.exists():
                log.warning("GeoNet metadata missing: %s", csv_path)
                continue
            df_year = pd.read_csv(csv_path, low_memory=False)
            df_year["__year"] = year
            frames.append(df_year)
        if not frames:
            raise FileNotFoundError(
                f"No metadata CSVs found in {self.events_csv_dir} "
                f"for years {self.years}"
            )
        df = pd.concat(frames, ignore_index=True)
        n_total = len(df)
        df = df[df["trace_category"] == "earthquake"].copy()

        # Audit GeoNet-A: explicit NaN-S drop with a single summary line.
        n_before_nan = len(df)
        df = df.dropna(subset=["trace_p_arrival_sample", "trace_s_arrival_sample"])
        n_dropped_nan = n_before_nan - len(df)
        if n_dropped_nan:
            log.info(
                "GeoNetSingleEQAdapter: dropped %d rows with NaN P/S picks "
                "(singleEQ contract requires both)",
                n_dropped_nan,
            )

        # Audit GeoNet-C: round() not int() on fractional picks.
        df["trace_p_arrival_sample"] = df["trace_p_arrival_sample"].astype(float).round().astype(int)
        df["trace_s_arrival_sample"] = df["trace_s_arrival_sample"].astype(float).round().astype(int)

        df = df[df["trace_s_arrival_sample"] > df["trace_p_arrival_sample"]]
        df = df[df["trace_p_arrival_sample"] > 0]
        df = df[df["trace_s_arrival_sample"] < self.original_npts]

        # Apply year-split assignment up-front; drop rows whose year is not
        # in any of the train/val/test tuples (e.g. 2013/2014 benchmark held-out).
        df["__split"] = df["__year"].map(
            lambda y: year_split(int(y), self.train_years, self.val_years, self.test_years)
        )
        n_held_out = int(df["__split"].isna().sum())
        if n_held_out:
            held_out_years = sorted(set(df.loc[df["__split"].isna(), "__year"]))
            log.info(
                "GeoNetSingleEQAdapter: skipping %d rows from held-out years %s",
                n_held_out, held_out_years,
            )
        df = df.dropna(subset=["__split"]).copy()

        log.info(
            "GeoNetSingleEQAdapter: %d EQ rows after %d filtered (catalog=%d)",
            len(df), n_total - len(df), n_total,
        )
        return df

    def __iter__(self) -> Iterator[WaveformSample]:
        df = self._load_metadata()
        rng = np.random.default_rng(self.seed)

        # Pre-compute per-row eligibility for the output window so we can
        # log a single summary line of drops (audit GeoNet-B) before the
        # heavy HDF5 reads start.
        ps_residual = (df["trace_s_arrival_sample"]
                       - df["trace_p_arrival_sample"]).astype(int)
        # Worst case: target_p == max_p_position → S lands at max_p_position + ps_residual.
        s_in_window = self.max_p_position + ps_residual
        too_long_mask = s_in_window >= (T_SAMPLES - self.min_back_buffer)
        n_too_long = int(too_long_mask.sum())
        if n_too_long:
            log.info(
                "GeoNetSingleEQAdapter: dropping %d rows whose P-S residual "
                "would push S past T_SAMPLES (=%d) - min_back_buffer (=%d)",
                n_too_long, T_SAMPLES, self.min_back_buffer,
            )
        df = df.loc[~too_long_mask].copy()

        # Group by year so we open each per-year HDF5 only once.
        n_yielded = 0
        for year, df_year in df.groupby("__year"):
            if self.max_samples is not None and n_yielded >= self.max_samples:
                break
            h5_path = self.events_hdf5_dir / f"waveforms_units_{year}.h5"
            if not h5_path.exists():
                log.warning("GeoNet waveform h5 missing: %s", h5_path)
                continue
            try:
                hf = h5py.File(h5_path, "r")
            except OSError as e:
                log.warning("Cannot open %s: %s", h5_path, e)
                continue

            with hf:
                data_grp = hf["data"]
                for _, row in df_year.iterrows():
                    if self.max_samples is not None and n_yielded >= self.max_samples:
                        break
                    trace_name = str(row["trace_name"])
                    split = str(row["__split"])
                    if self.split_filter and split != self.split_filter:
                        continue
                    p_sample = int(row["trace_p_arrival_sample"])
                    s_sample = int(row["trace_s_arrival_sample"])
                    if not (0 < p_sample < s_sample < self.original_npts):
                        continue
                    ps_residual_row = s_sample - p_sample

                    try:
                        arr = np.array(data_grp[trace_name])
                    except (KeyError, OSError):
                        continue

                    arr32 = _coerce_3x12000(arr, self.original_npts)
                    if arr32 is None:
                        continue

                    sample = self._slice_or_pad(
                        arr32, p_sample, s_sample, rng,
                    )
                    if sample is None:
                        continue
                    wf_cf, new_p, new_s = sample
                    if not (0 <= new_p < T_SAMPLES and 0 <= new_s < T_SAMPLES):
                        continue
                    if not np.isfinite(wf_cf).all():
                        continue

                    # Sample-id includes year so two traces with identical
                    # ``trace_name`` from different years (rare but possible
                    # under cross-year station naming) won't collide either
                    # in the H5 index or in any future padded_cache.
                    yield WaveformSample(
                        sample_id=f"singleEQ_{year}_{trace_name}",
                        waveform=wf_cf,
                        category="singleEQ",
                        sampling_category=_ps_diff_bin(ps_residual_row),
                        split=split,
                        source_file=str(h5_path),
                        p_arrival_samples=[int(new_p)],
                        s_arrival_samples=[int(new_s)],
                        polarity="",   # GeoNet has no polarity labels
                    )
                    n_yielded += 1

    # ---------------------------------------------------------------- helpers
    def _slice_or_pad(
        self,
        arr32: np.ndarray,
        p_sample: int,
        s_sample: int,
        rng: np.random.Generator,
    ) -> Optional[tuple[np.ndarray, int, int]]:
        """Position P at a random offset within the 9000-sample output.

        Strategy:
            1. Pick target_p in [min_p_position, max_p_position].
            2. slice_start = p_sample - target_p; slice_end = slice_start + T.
            3. If [slice_start, slice_end) fits inside [0, original_npts), pure slice.
            4. Otherwise, pad spectrum-matched noise on the deficient side(s)
               using pre-P signal as reference for the front pad and the
               trace tail as reference for the back pad (mirrors legacy P01).
        """
        ps_residual = s_sample - p_sample
        target_p = int(rng.integers(self.min_p_position, self.max_p_position + 1))
        slice_start = p_sample - target_p
        slice_end = slice_start + T_SAMPLES
        npts = arr32.shape[1]

        if 0 <= slice_start and slice_end <= npts:
            # Pure slice, no padding needed.
            wf = arr32[:, slice_start:slice_end].copy()
            return wf, target_p, target_p + ps_residual

        # Pad-and-slice via spectrum-matched noise on the deficient sides.
        front_deficit = max(0, -slice_start)
        back_deficit = max(0, slice_end - npts)

        # Build references per channel and synthesize matching noise.
        out = np.zeros((N_CHANNELS, T_SAMPLES), dtype=np.float32)
        for ch in range(N_CHANNELS):
            data_ch = arr32[ch]
            # Real portion that survives the slice
            real_start = max(0, slice_start)
            real_end = min(npts, slice_end)
            real_seg = data_ch[real_start:real_end]

            # Front reference: pre-P samples
            pre_p = data_ch[:p_sample] if p_sample > 0 else data_ch[:500]
            if len(pre_p) >= 500:
                front_ref = find_reference_signal(
                    pre_p, window_size=500,
                    max_search=len(pre_p), min_unique=100,
                )
            elif len(pre_p) >= 200:
                front_ref = find_reference_signal(
                    pre_p, window_size=200,
                    max_search=len(pre_p), min_unique=50,
                )
            else:
                front_ref = pre_p if len(pre_p) > 0 else data_ch[:500]

            # Back reference: trace tail (away from S coda where possible)
            tail_start = max(s_sample + 100, npts - 500)
            tail = data_ch[tail_start:]
            back_ref = tail if len(tail) >= 200 else front_ref

            # Synthesise pads only where needed.
            if front_deficit > 0:
                front_noise = generate_matching_noise(
                    front_ref, front_deficit
                ).astype(np.float32)
            else:
                front_noise = np.empty(0, dtype=np.float32)
            if back_deficit > 0:
                back_noise = generate_matching_noise(
                    back_ref, back_deficit
                ).astype(np.float32)
            else:
                back_noise = np.empty(0, dtype=np.float32)

            seg = np.concatenate([front_noise, real_seg, back_noise])
            if seg.shape[0] != T_SAMPLES:
                # Length mismatch shouldn't happen by construction, but guard.
                return None
            out[ch] = seg

        # P sample now sits at: front_deficit + (p_sample - real_start)
        new_p = front_deficit + (p_sample - max(0, slice_start))
        new_s = new_p + ps_residual
        return out, new_p, new_s
