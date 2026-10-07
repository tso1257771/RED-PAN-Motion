"""OBSTransformer singleEQ adapter reading from training_data.hdf5.

Source layout:

    <obst_root>/
        training_data.hdf5    # /data/{trace_name}: (6000, 3) float32 channel-last
        P01_TFRecord_singleEQ.py     # legacy 90 s TFRecord generator (reference only)

Per-dataset HDF5 has 60 394 entries:
    * 35 394 entries with suffix ``_EV`` — earthquake event traces (used here)
    * 25 000 entries with suffix ``_NO`` — noise traces (intentionally skipped)

Each ``_EV`` dataset is shape ``(6000, 3)`` float32 channel-last, channel order
(E, N, Z) per the STEAD-derived format. Each carries attrs:
    - ``p_arrival_sample``  : int   (within the 6000-sample trace)
    - ``s_arrival_sample``  : int   (within the 6000-sample trace)
    - ``snr_db``            : (3,) float
    - ``trace_category``    : str   ('earthquake_local')
    - ``trace_name``        : str   (mirror of dict key)

Pad/window strategy for the unified 90 s (9000-sample) target:
    1. Read the (6000, 3) trace channel-last.
    2. Build a generous spectrum-matched buffer with front-pad reference taken
       from the pre-P window of the trace itself, and back-pad reference from
       the post-S coda. (Mirrors the legacy ``P01_TFRecord_singleEQ.py`` logic.)
    3. Random-slice a 9000-sample window with P landing at a uniformly random
       position in ``[min_p_position, max_p_position]``.
    4. Transpose to channel-first (E, N, Z) for emission.

OBSTransformer has no first-motion polarity labels — ``polarity`` stays ''.
The category is always ``singleEQ``.

Audit fixes applied (2026-04-27):

* 3-way split via :func:`hash_split` keyed on the event identifier (the
  trace_name with the trailing ``_EV`` stripped — multiple stations
  recording the same event share that prefix). Replaces the previous
  per-trace shuffled Bernoulli (train+val only).
* ``int(round(...))`` instead of ``int(...)`` when reading
  ``p_arrival_sample`` / ``s_arrival_sample`` HDF5 attrs (some attrs are
  stored as float in the source).
* Tightened ``max_target_p`` clamp so the random slice cannot place
  ``new_s = target_p + ps_residual`` past ``output_npts``. The legacy
  P01 generator's ``max_front_pad = max(max_front_pad, min_front_pad +
  100)`` rebound allowed traces with ``s_sample`` near the trace end to
  emit ``new_s >= output_npts``. We now SKIP the row when the per-row
  upper bound on ``target_p`` falls below ``min_p_position``.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from .base import BaseAdapter
from ..schema import (
    WaveformSample, T_SAMPLES, N_CHANNELS, SAMPLING_RATE_HZ,
)
from ..splits import hash_split
from redpan_motion.utils.waveform import find_reference_signal, generate_matching_noise


log = logging.getLogger(__name__)


_SOURCE_NPTS = 6000   # OBSTransformer training_data.hdf5: 60 s @ 100 Hz


def _ps_diff_bin(ps_diff_samples: int) -> str:
    """Match train_*.json ``category_weights`` bin keys."""
    sec = ps_diff_samples / SAMPLING_RATE_HZ
    if sec < 5:   return "singleEQ_00-05s"
    if sec < 10:  return "singleEQ_05-10s"
    if sec < 15:  return "singleEQ_10-15s"
    if sec < 20:  return "singleEQ_15-20s"
    return "singleEQ_20s_plus"


def _event_id_from_trace(trace_name: str) -> str:
    """Strip the ``_EV`` / ``_NO`` suffix to expose the event identifier.

    OBSTransformer trace_names look like
    ``<station>.<network>_<event_timestamp>_EV`` — every station recording
    the same event shares the prefix without ``_EV``, so using that prefix
    as the split key prevents same-event leakage across train/val/test.
    """
    if trace_name.endswith("_EV") or trace_name.endswith("_NO"):
        return trace_name[:-3]
    return trace_name


class OBSTransformerSingleEQAdapter(BaseAdapter):
    """OBSTransformer single-event: spectrum-matched pad + random P-position slice.

    Args:
        training_hdf5    : path to ``training_data.hdf5``.
        p_train          : fraction routed to the ``train`` split via
                           :func:`hash_split` (default 0.70).
        p_val            : fraction routed to the ``val`` split (default
                           0.15); remainder becomes ``test``.
        seed             : RNG seed for window-shift offsets only — split
                           assignment is fully deterministic via the salt.
        min_p_position   : lower bound (inclusive) for P sample index in the
                           emitted 9000-sample window.
        max_p_position   : upper bound (inclusive). Per-row this is further
                           tightened so the matching S-pick stays inside
                           ``[0, T_SAMPLES)``.
        split_salt       : salt for the deterministic hash-split.

    Notes:
        Splits are deterministic: given the same salt, every trace from the
        same source event lands in the same split across runs. The RNG is
        used only for in-window shift offsets.
    """

    name = "OBSTransformer"
    category = "singleEQ"

    def __init__(
        self,
        training_hdf5: Path,
        *,
        p_train: float = 0.70,
        p_val: float = 0.15,
        seed: int = 42,
        min_p_position: int = 500,
        max_p_position: int = 7500,
        split_salt: str = "OBSTransformer-singleEQ-v1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.training_hdf5 = Path(training_hdf5)
        self.p_train = p_train
        self.p_val = p_val
        self.seed = seed
        self.min_p_position = min_p_position
        self.max_p_position = max_p_position
        self.split_salt = split_salt

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_padded_buffer(
        self,
        arr: np.ndarray,
        p_sample: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        """Spectrum-matched two-sided pad around the source trace.

        Builds a buffer wide enough to support any P position in
        ``[min_p_position, max_p_position]`` after random slicing.

        Returns:
            padded   : (front_pad_n + npts + back_pad_n, 3) float32 channel-last.
            front_pad_n : number of samples prepended (P position offset basis).
        """
        npts, n_ch = arr.shape   # (6000, 3)
        # Pad budget: large enough so that any random P position fits.
        front_pad_n = self.max_p_position + 1000
        back_pad_n  = T_SAMPLES + 1000

        padded = np.zeros((front_pad_n + npts + back_pad_n, n_ch), dtype=np.float32)

        for ch in range(n_ch):
            data = arr[:, ch]

            # Front reference: pre-P window (clean of phase content).
            pre_p_end = min(max(p_sample, 1), len(data))
            pre_p = data[:pre_p_end]
            if len(pre_p) >= 500:
                front_ref = find_reference_signal(
                    pre_p, window_size=500, max_search=len(pre_p), min_unique=100,
                )
            elif len(pre_p) >= 200:
                front_ref = find_reference_signal(
                    pre_p, window_size=200, max_search=len(pre_p), min_unique=50,
                )
            else:
                front_ref = pre_p if len(pre_p) > 0 else data[:500]

            # Back reference: post-S coda tail, fallback to front_ref.
            tail_length = min(500, len(data))
            back_ref = data[-tail_length:] if tail_length >= 200 else front_ref

            front_noise = generate_matching_noise(front_ref, front_pad_n).astype(np.float32)
            back_noise  = generate_matching_noise(back_ref,  back_pad_n).astype(np.float32)

            padded[:, ch] = np.concatenate([front_noise, data, back_noise])

        return padded, front_pad_n

    def __iter__(self) -> Iterator[WaveformSample]:
        log.info("OBSTransformerSingleEQAdapter: reading %s", self.training_hdf5)

        with h5py.File(self.training_hdf5, "r") as hf:
            data_grp = hf.get("data")
            if data_grp is None:
                raise ValueError(f"missing 'data' group in {self.training_hdf5}")

            all_keys = [k for k in data_grp.keys() if k.endswith("_EV")]
            log.info("  EV traces in HDF5: %d", len(all_keys))

            # Stable order so smoke-test outputs are reproducible. The RNG
            # is reserved for in-window shifts only.
            sorted_keys = sorted(all_keys)
            rng = np.random.default_rng(self.seed)

            n_yielded = 0
            n_dropped_clamp = 0
            for trace_name in sorted_keys:
                if self.max_samples is not None and n_yielded >= self.max_samples:
                    break

                event_key = _event_id_from_trace(trace_name)
                split = hash_split(
                    event_key, self.p_train, self.p_val, salt=self.split_salt,
                )
                if self.split_filter and split != self.split_filter:
                    continue

                ds = data_grp.get(trace_name)
                if ds is None:
                    continue

                # int(round(float(...))) — attrs may be stored as float in
                # the source HDF5; legacy int(...) truncated toward zero.
                try:
                    p_sample = int(round(float(ds.attrs["p_arrival_sample"])))
                    s_sample = int(round(float(ds.attrs["s_arrival_sample"])))
                except (KeyError, TypeError, ValueError):
                    continue
                if not (0 < p_sample < s_sample < _SOURCE_NPTS):
                    continue
                ps_residual = s_sample - p_sample

                # AUDIT FIX: tighten the upper bound on the random target P
                # position so new_s = target_p + ps_residual stays strictly
                # inside [0, T_SAMPLES). If the per-row upper bound falls
                # below min_p_position, skip the row rather than clamping
                # back up — clamping was the legacy bug that allowed
                # new_s >= output_npts for traces with s_sample near the end.
                effective_max_target_p = min(
                    self.max_p_position, T_SAMPLES - 1 - ps_residual,
                )
                if effective_max_target_p < self.min_p_position:
                    n_dropped_clamp += 1
                    continue

                try:
                    arr = np.asarray(ds, dtype=np.float32)
                except (KeyError, OSError):
                    continue
                if arr.shape != (_SOURCE_NPTS, N_CHANNELS):
                    log.debug("skip %s: unexpected shape %s", trace_name, arr.shape)
                    continue

                # Two-sided spectrum-matched pad to support any P-shift.
                padded, front_pad_n = self._build_padded_buffer(arr, p_sample, rng)

                # Random P position inside the 9000-sample output window,
                # using the per-row clamped upper bound.
                target_p = int(rng.integers(
                    self.min_p_position, effective_max_target_p + 1,
                ))
                p_in_padded = front_pad_n + p_sample
                slice_start = p_in_padded - target_p
                slice_end = slice_start + T_SAMPLES
                if slice_start < 0 or slice_end > padded.shape[0]:
                    continue

                sliced = padded[slice_start:slice_end]   # (9000, 3)
                new_p = target_p
                new_s = target_p + ps_residual
                # Defensive double-check (clamp above should already enforce):
                if not (0 <= new_p < T_SAMPLES and 0 <= new_s < T_SAMPLES):
                    continue

                wf_cf = sliced.T.astype(np.float32, copy=False).copy()
                if not np.isfinite(wf_cf).all():
                    log.debug("skip %s: non-finite after pad", trace_name)
                    continue

                yield WaveformSample(
                    sample_id=f"singleEQ_{trace_name}",
                    waveform=wf_cf,
                    category="singleEQ",
                    sampling_category=_ps_diff_bin(ps_residual),
                    split=split,  # type: ignore[arg-type]
                    source_file=str(self.training_hdf5),
                    p_arrival_samples=[int(new_p)],
                    s_arrival_samples=[int(new_s)],
                    polarity="",   # OBSTransformer carries no first-motion labels
                )
                n_yielded += 1

            if n_dropped_clamp:
                log.info(
                    "OBSTransformerSingleEQAdapter: skipped %d rows where "
                    "s_sample was too close to source end "
                    "(post-clamp max_target_p < min_p_position).",
                    n_dropped_clamp,
                )
