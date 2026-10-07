"""STEAD noise + singleEQ adapters reading directly from merge.hdf5.

Replicates the (noise) front-pad logic of the legacy `P05_TFRecord_noise.py`
and (singleEQ) the random-slice + both-side spectrum-matched pad of
`P01_TFRecord_singleEQ.py`, but emits WaveformSample directly — no TFRecord.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Iterator, Optional, Literal
import logging

import h5py
import numpy as np
import pandas as pd

from .base import BaseAdapter
from ..schema import (
    WaveformSample, T_SAMPLES, N_CHANNELS, SAMPLING_RATE_HZ,
)
from ..splits import hash_split, bernoulli_3way
# Spectrum-matched padding helpers (identical to the RED-PAN redpan/utils.py versions).
from redpan_motion.utils.waveform import find_reference_signal, generate_matching_noise


log = logging.getLogger(__name__)


def _ps_diff_bin(ps_diff_samples: int) -> str:
    """Convert P-S sample difference into a sampling_category bin name
    matching the train_*.json category_weights keys."""
    sec = ps_diff_samples / SAMPLING_RATE_HZ
    if sec < 5:   return "singleEQ_00-05s"
    if sec < 10:  return "singleEQ_05-10s"
    if sec < 15:  return "singleEQ_10-15s"
    if sec < 20:  return "singleEQ_15-20s"
    return "singleEQ_20s_plus"


class STEADNoiseAdapter(BaseAdapter):
    """STEAD noise: 60 s real → 90 s with 30 s spectrum-matched front pad.

    Pad reference is the first 500 non-flat samples of the trace itself, then
    `generate_matching_noise(reference, 3000)` is concatenated at the front.
    Labels: empty p/s. Output category = 'noise'.

    ``exclude_traces`` (e.g. the names in STEAD's ``test.npy``) are dropped before
    the split draw, so the official STEAD test noise never enters this archive;
    the benchmark builds it separately (``build_stead_noise_test.py``).
    """
    name = "STEAD"
    category = "noise"

    def __init__(
        self,
        merge_hdf5: Path,
        merge_csv: Path,
        *,
        p_train: float = 0.70,
        p_val: float = 0.15,
        seed: int = 42,
        exclude_traces: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.merge_hdf5 = Path(merge_hdf5)
        self.merge_csv = Path(merge_csv)
        self.p_train = p_train
        self.p_val = p_val
        self.seed = seed
        self.exclude_traces = set(exclude_traces) if exclude_traces is not None else set()

    def __iter__(self) -> Iterator[WaveformSample]:
        log.info("STEADNoiseAdapter: reading %s", self.merge_csv)
        df = pd.read_csv(self.merge_csv, low_memory=False)
        nz = df[df["trace_category"] == "noise"].copy()
        log.info("  noise rows in merge.csv: %d", len(nz))
        if self.exclude_traces:
            nz = nz[~nz["trace_name"].astype(str).isin(self.exclude_traces)]
            log.info("  after excluding %d listed traces: %d", len(self.exclude_traces), len(nz))

        # Noise samples are independent, so per-row Bernoulli is fine.
        rng = np.random.default_rng(self.seed)

        n_yielded = 0
        with h5py.File(self.merge_hdf5, "r") as hf:
            for _, row in nz.iterrows():
                if self.max_samples is not None and n_yielded >= self.max_samples:
                    break
                trace_name = str(row["trace_name"])
                split = bernoulli_3way(rng, self.p_train, self.p_val)
                if self.split_filter and split != self.split_filter:
                    continue

                try:
                    arr = np.array(hf["data"][trace_name]).astype(np.float32)
                except (KeyError, OSError):
                    continue

                # STEAD waveforms in merge.hdf5 are (6000, 3) channel-last.
                if arr.shape != (6000, 3):
                    continue

                # Front-pad 30 s of spectrum-matched noise per channel.
                pad_npts = T_SAMPLES - arr.shape[0]   # 9000 - 6000 = 3000
                padded = np.zeros((T_SAMPLES, N_CHANNELS), dtype=np.float32)
                for ch in range(N_CHANNELS):
                    ref = find_reference_signal(
                        arr[:, ch], window_size=500, max_search=5000, min_unique=300,
                    )
                    front = generate_matching_noise(ref, pad_npts).astype(np.float32)
                    padded[:pad_npts, ch] = front
                    padded[pad_npts:, ch] = arr[:, ch]

                # Channel-first (E, N, Z): STEAD's column order in merge.hdf5
                # is documented as (E, N, Z) so this is a transpose only.
                wf_cf = padded.T.astype(np.float32, copy=False).copy()
                # Non-finite guard (find_reference_signal can produce edge NaN)
                if not np.isfinite(wf_cf).all():
                    log.warning("skipping %s: non-finite after pad", trace_name)
                    continue

                yield WaveformSample(
                    sample_id=f"noise_{trace_name}",
                    waveform=wf_cf,
                    category="noise",
                    sampling_category="noise",
                    split=split,
                    source_file=str(self.merge_hdf5),
                    p_arrival_samples=[],
                    s_arrival_samples=[],
                    polarity="",
                )
                n_yielded += 1


class STEADSingleEQAdapter(BaseAdapter):
    """STEAD single-event: pad both sides spectrum-matched, random-slice so
    P lands at random offset within the 9000-sample window."""
    name = "STEAD"
    category = "singleEQ"

    def __init__(
        self,
        merge_hdf5: Path,
        merge_csv: Path,
        *,
        p_train: float = 0.70,
        p_val: float = 0.15,
        seed: int = 42,
        min_p_position: int = 500,
        max_p_position: int = 7500,
        split_salt: str = "STEAD-singleEQ-v1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.merge_hdf5 = Path(merge_hdf5)
        self.merge_csv = Path(merge_csv)
        self.p_train = p_train
        self.p_val = p_val
        self.seed = seed
        self.min_p_position = min_p_position
        self.max_p_position = max_p_position
        self.split_salt = split_salt

    def __iter__(self) -> Iterator[WaveformSample]:
        log.info("STEADSingleEQAdapter: reading %s", self.merge_csv)
        df = pd.read_csv(self.merge_csv, low_memory=False)
        eq = df[df["trace_category"] == "earthquake_local"].copy()
        log.info("  EQ rows in merge.csv: %d", len(eq))

        rng = np.random.default_rng(self.seed)

        n_yielded = 0
        with h5py.File(self.merge_hdf5, "r") as hf:
            for _, row in eq.iterrows():
                if self.max_samples is not None and n_yielded >= self.max_samples:
                    break
                trace_name = str(row["trace_name"])
                # Event-aware split: every trace from the same source_id lands
                # in the same split — prevents same-event leakage between
                # train/val/test. Falls back to trace_name if source_id absent.
                split_key = str(row.get("source_id", "") or trace_name)
                split = hash_split(split_key, self.p_train, self.p_val,
                                   salt=self.split_salt)
                if self.split_filter and split != self.split_filter:
                    continue
                try:
                    # Bug fix: use round() instead of int() to avoid systematic
                    # negative-sample bias from truncating fractional picks.
                    p_sample = int(round(float(row["p_arrival_sample"])))
                    s_sample = int(round(float(row["s_arrival_sample"])))
                except (TypeError, ValueError):
                    continue
                if not (0 < p_sample < s_sample < 6000):
                    continue
                ps_residual = s_sample - p_sample

                try:
                    arr = np.array(hf["data"][trace_name]).astype(np.float32)
                except (KeyError, OSError):
                    continue
                if arr.shape != (6000, 3):
                    continue

                # Pad both sides spectrum-matched per the legacy P01 logic
                front_pad_n = self.max_p_position + 1000
                back_pad_n  = T_SAMPLES + 1000
                npts = arr.shape[0]
                padded = np.zeros((front_pad_n + npts + back_pad_n, N_CHANNELS), dtype=np.float32)
                for ch in range(N_CHANNELS):
                    pre_p = arr[:p_sample, ch] if p_sample > 0 else arr[:500, ch]
                    front_ref = (find_reference_signal(pre_p, window_size=500, max_search=len(pre_p), min_unique=100)
                                 if len(pre_p) >= 500 else pre_p)
                    tail = arr[max(0, npts - 500):, ch]
                    back_ref = tail if len(tail) >= 200 else front_ref
                    front_noise = generate_matching_noise(front_ref, front_pad_n).astype(np.float32)
                    back_noise = generate_matching_noise(back_ref, back_pad_n).astype(np.float32)
                    padded[:, ch] = np.concatenate([front_noise, arr[:, ch], back_noise])

                # Random P position in output
                p_in_padded = front_pad_n + p_sample
                target_p = int(rng.integers(self.min_p_position, self.max_p_position + 1))
                slice_start = p_in_padded - target_p
                slice_end = slice_start + T_SAMPLES
                if slice_start < 0 or slice_end > padded.shape[0]:
                    continue
                sliced = padded[slice_start:slice_end]   # (9000, 3)

                new_p = target_p
                new_s = target_p + ps_residual
                if not (0 <= new_p < T_SAMPLES and 0 <= new_s < T_SAMPLES):
                    continue

                wf_cf = sliced.T.astype(np.float32, copy=False).copy()
                if not np.isfinite(wf_cf).all():
                    continue

                yield WaveformSample(
                    sample_id=f"singleEQ_{trace_name}",
                    waveform=wf_cf,
                    category="singleEQ",
                    sampling_category=_ps_diff_bin(ps_residual),
                    split=split,
                    source_file=str(self.merge_hdf5),
                    p_arrival_samples=[int(new_p)],
                    s_arrival_samples=[int(new_s)],
                    polarity="",   # STEAD has no first-motion labels
                )
                n_yielded += 1
