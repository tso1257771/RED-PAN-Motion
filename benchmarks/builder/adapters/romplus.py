"""ROMPLUS singleEQ + Ponly + Sonly adapters — SAC reading with per-channel
alignment validation.

Source layout (raw):

    <romplus_root>/sac/{year}/{event_id}/{net}.{sta}.{loc}.{ch}{ENZ|12Z}.sac
    <romplus_root>/romplus_metadata.csv
        — columns: year, event, network, station, location, channel,
          station_id, tp_utc, ts_utc, psres, snr, pick_source_p, pick_source_s

ROMPLUS spans Romanian (RO) and Italian (IV/MN/GE/...) networks; channels can be
``BHE/BHN/BHZ`` (geographic E/N) or ``HH1/HH2/HHZ`` / ``EH1/EH2/EHZ``
(horizontal pair without geographic alignment). The Z component is well-defined
in either case, so 12Z rows are kept (only horizontals are rotated).

Audit-driven validators (per the legacy P0*-script bug analysis):

* **Sampling rate** — verify *all 3 channels* are at 100 Hz (legacy generator
  only checked ``wf[0]``; if ch 2 was at 50 Hz, every pick label index was
  wrong by a factor of 2 on that channel).
* **Component length** — verify ``len(wf[0]) == len(wf[1]) == len(wf[2])``
  (legacy only checked ``wf[0]``).
* **Inter-component starttime drift** — verify
  ``max(|starttime_i − starttime_0|) < 1 sample`` so the single pick index is
  valid for every channel (legacy used ``wf[0].stats.starttime`` for all three).
* **SEED-suffix-aware channel sort** — explicit map ``E/1 → 0, N/2 → 1, Z → 2``
  rather than alphabetical (``HH1<HH2<HHZ`` would otherwise misalign).
* **Pick-time bound** — pre-P window must fit before the slice front-pad, and
  post-S coda must fit before the slice back-pad.

3-way deterministic split via :func:`hash_split` keyed on ``year/event_id`` so
all stations of the same earthquake land in the same split.

Picks parsed via ObsPy ``UTCDateTime`` and converted to integer sample indices
with banker's rounding (``int(round(seconds_offset * sampling_rate))``). No
``int(...)`` truncation bias.

Concrete adapters
-----------------
* :class:`ROMPLUSSingleEQAdapter` — both P and S in window (default).
* :class:`ROMPLUSSingleEQZeropadAdapter` — singleEQ + horizontal channel-drop.
* :class:`ROMPLUSPonlyAdapter` — window contains P, S deliberately outside.
* :class:`ROMPLUSSonlyAdapter` — window contains S, P deliberately outside.

The Ponly/Sonly variants are derived from the same SAC catalogue as singleEQ
(both picks always present in metadata). They differ only in *where the window
is placed* relative to the picks, mirroring the CREW Ponly/Sonly slicing
strategy.
"""
from __future__ import annotations
import logging
from glob import glob
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseAdapter
from ..schema import (
    WaveformSample, T_SAMPLES, N_CHANNELS, SAMPLING_RATE_HZ,
)
from ..splits import hash_split
from redpan_motion.utils.waveform import find_reference_signal, generate_matching_noise


log = logging.getLogger(__name__)


# Ordering map for SEED channel last-letter — drives the (E, N, Z) → (0, 1, 2)
# placement regardless of alphabetical sort. ``1`` and ``2`` are accepted for
# 12Z stations and placed in the horizontal slots; the model is then trained
# with rotated horizontals on those rows but the Z channel is correct.
_SUFFIX_TO_INDEX = {
    "E": 0, "1": 0,
    "N": 1, "2": 1,
    "Z": 2,
}


def _ps_diff_bin(ps_diff_samples: int) -> str:
    """sampling_category bin matching the train_*.json category_weights keys."""
    sec = ps_diff_samples / SAMPLING_RATE_HZ
    if sec < 5:   return "singleEQ_00-05s"
    if sec < 10:  return "singleEQ_05-10s"
    if sec < 15:  return "singleEQ_10-15s"
    if sec < 20:  return "singleEQ_15-20s"
    return "singleEQ_20s_plus"


def _read_three_channel_sac(
    sac_root: Path,
    year: str, event: str, net: str, sta: str, loc: str, channel: str,
):
    """Read 3 SAC components for one station-event. Return (3, npts) float32 +
    (utc_starttime, sampling_rate) on success, or None on any failure.

    Channels searched in priority order:
      ``{ch}E + {ch}N + {ch}Z`` (geographic)  →  fall back to
      ``{ch}1 + {ch}2 + {ch}Z`` (12Z station)
    """
    from obspy import read

    base = sac_root / year / event
    loc_str = "" if loc in ("", "nan", "NaN") else loc
    fname_pat = lambda ch_suffix: f"{net}.{sta}.{loc_str}.{channel}{ch_suffix}.sac"

    triples = [("E", "N", "Z"), ("1", "2", "Z")]
    for triple in triples:
        paths = [base / fname_pat(s) for s in triple]
        if all(p.exists() for p in paths):
            try:
                streams = [read(str(p))[0] for p in paths]
            except Exception as e:
                log.debug("SAC read failed for %s: %s", paths, e)
                return None
            return triple, streams
    return None


def _validate_and_align(triple, streams, *,
                        max_drift_samples: float = 1.0
                        ) -> Optional[tuple[np.ndarray, "UTCDateTime"]]:
    """Apply the per-component validators. Return (3, npts) float32 array
    in (E/1, N/2, Z) order plus a single starttime for label arithmetic, or
    None on any check failure."""
    # 1. Sampling rate — every channel must be 100 Hz
    for st in streams:
        if int(round(float(st.stats.sampling_rate))) != SAMPLING_RATE_HZ:
            return None

    # 2. Length — all three must agree exactly
    npts0 = streams[0].stats.npts
    if not all(st.stats.npts == npts0 for st in streams):
        return None

    # 3. Inter-component starttime drift
    t0 = streams[0].stats.starttime
    dt = 1.0 / SAMPLING_RATE_HZ
    drifts = [abs(float(st.stats.starttime - t0)) for st in streams]
    if max(drifts) > max_drift_samples * dt:
        return None

    # 4. Order channels via the SEED-suffix map
    arr = np.zeros((N_CHANNELS, npts0), dtype=np.float32)
    for suffix, st in zip(triple, streams):
        idx = _SUFFIX_TO_INDEX.get(suffix)
        if idx is None:
            return None
        arr[idx, :] = st.data.astype(np.float32, copy=False)

    if not np.isfinite(arr).all():
        return None
    return arr, t0


# Type alias: per-row slice plan returned by ``_compute_slice_plan``.
#   (slice_start, p_in_window_list, s_in_window_list, sampling_category)
SlicePlan = Tuple[int, List[int], List[int], str]


class _ROMPLUSBaseAdapter(BaseAdapter):
    """Shared SAC reader + windower for ROMPLUS variants.

    The default ``_compute_slice_plan`` implements the singleEQ slicing logic
    (random target_p offset, ps_residual binning). Subclasses override
    ``_compute_slice_plan`` to produce Ponly / Sonly partial-pick windows
    while reusing the SAC-reading and spectrum-matched padding pipeline.
    """

    name = "ROMPLUS"
    category = "singleEQ"
    split_salt = "ROMPLUS-singleEQ-v1"

    # Margin (samples) keeping the excluded phase strictly outside the window
    # for Ponly / Sonly. Mirrors the CREW partial-pick logic.
    PHASE_EXCLUSION_MARGIN = 200  # 2 s @ 100 Hz

    def __init__(
        self,
        sac_root: Path,
        metadata_csv: Path,
        *,
        p_train: float = 0.70,
        p_val: float = 0.15,
        seed: int = 42,
        min_p_position: int = 1000,
        max_p_position: int = 7000,
        max_drift_samples: float = 1.0,
        split_salt: Optional[str] = None,
        stratify_ps_bins: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sac_root = Path(sac_root)
        self.metadata_csv = Path(metadata_csv)
        self.p_train = p_train
        self.p_val = p_val
        self.seed = seed
        self.min_p_position = min_p_position
        self.max_p_position = max_p_position
        self.max_drift_samples = max_drift_samples
        if split_salt is not None:
            # Allow the caller to override the per-class default (used by
            # ``ROMPLUSSingleEQZeropadAdapter`` and a few one-off tests).
            self.split_salt = split_salt
        # When True, cap per-bin yield at ceil(max_samples/5) so the output
        # is balanced across the 5 sampling_category P-S residual bins.
        # Only meaningful for singleEQ_zeropad-style runs that have a real
        # P-S residual; Ponly / Sonly samp_cats fall outside the bin keys
        # and bypass the stratification check at the iter site.
        self.stratify_ps_bins = stratify_ps_bins

    # -- subclass hooks --------------------------------------------------
    def _compute_slice_plan(
        self,
        tp_npts: int,
        ts_npts: int,
        full_npts: int,
        rng: np.random.Generator,
    ) -> Optional[SlicePlan]:
        """Decide where the 9 000-sample window starts and which picks are
        retained for one metadata row. Default: the singleEQ strategy.

        Returns (slice_start, p_in_window_list, s_in_window_list,
        sampling_category) or None to skip the row.
        """
        ps_residual = ts_npts - tp_npts
        if not (0 < ps_residual < T_SAMPLES - self.min_p_position):
            return None

        target_p = int(rng.integers(self.min_p_position,
                                    self.max_p_position + 1))
        slice_start = tp_npts - target_p

        new_p = target_p
        new_s = target_p + ps_residual
        if not (0 <= new_p < T_SAMPLES and 0 <= new_s < T_SAMPLES):
            return None

        return slice_start, [new_p], [new_s], _ps_diff_bin(ps_residual)

    def _post_window(
        self,
        wf: np.ndarray,
        p_list: List[int],
        s_list: List[int],
        rng: np.random.Generator,
    ) -> Optional[tuple[np.ndarray, List[int], List[int], str]]:
        """Optionally mutate the (3, T_SAMPLES) window. Default: identity.

        Subclasses may return ``None`` to drop the sample, or a 4-tuple
        ``(wf, p_list, s_list, category)`` where ``category`` is the value
        written into ``WaveformSample.category`` (typically ``self.category``).
        """
        return wf, p_list, s_list, self.category

    # -- iterator --------------------------------------------------------
    def __iter__(self) -> Iterator[WaveformSample]:
        from obspy import UTCDateTime

        log.info("ROMPLUS%s: reading %s", self.category, self.metadata_csv)
        df = pd.read_csv(self.metadata_csv, low_memory=False)
        log.info("  %d catalogue rows", len(df))

        rng = np.random.default_rng(self.seed)
        n_yielded = 0
        n_skip_sac, n_skip_align, n_skip_pick = 0, 0, 0
        n_skip_plan = 0
        n_skip_bin_full = 0

        # Stratification setup: per-bin cap so the output is evenly spread
        # across the 5 sampling_category P-S residual bins.
        bin_caps: dict = {}
        bin_counts: dict = {}
        if self.stratify_ps_bins and self.max_samples:
            n_bins = 5
            cap = max(1, (self.max_samples + n_bins - 1) // n_bins)
            for b in ("singleEQ_00-05s", "singleEQ_05-10s",
                      "singleEQ_10-15s", "singleEQ_15-20s",
                      "singleEQ_20s_plus"):
                bin_caps[b] = cap
                bin_counts[b] = 0
            log.info("ROMPLUS%s: stratify_ps_bins=True, per-bin cap=%d",
                     self.category, cap)

        for _, row in df.iterrows():
            if self.max_samples is not None and n_yielded >= self.max_samples:
                break

            year = str(int(row["year"]))
            event = str(row["event"]).zfill(7)
            net = str(row["network"])
            sta = str(row["station"])
            loc = "" if pd.isna(row.get("location")) else str(row["location"]).strip()
            channel = str(row["channel"])     # 'BH' / 'HH' / 'EH' ...

            # Event-aware split: same earthquake → same split, all stations.
            split_key = f"{year}_{event}"
            split = hash_split(split_key, self.p_train, self.p_val,
                               salt=self.split_salt)
            if self.split_filter and split != self.split_filter:
                continue

            res = _read_three_channel_sac(
                self.sac_root, year, event, net, sta, loc, channel
            )
            if res is None:
                n_skip_sac += 1
                continue
            triple, streams = res

            ali = _validate_and_align(
                triple, streams, max_drift_samples=self.max_drift_samples
            )
            if ali is None:
                n_skip_align += 1
                continue
            full_wf, t0 = ali

            # Pick conversion: tp/ts UTC strings → samples (banker's round)
            try:
                tp_utc = UTCDateTime(str(row["tp_utc"]))
                ts_utc = UTCDateTime(str(row["ts_utc"]))
            except Exception:
                n_skip_pick += 1
                continue
            tp_npts = int(round(float(tp_utc - t0) * SAMPLING_RATE_HZ))
            ts_npts = int(round(float(ts_utc - t0) * SAMPLING_RATE_HZ))
            if not (0 <= tp_npts < full_wf.shape[1]
                    and 0 <= ts_npts < full_wf.shape[1]):
                n_skip_pick += 1
                continue

            # Per-category window placement.
            plan = self._compute_slice_plan(
                tp_npts, ts_npts, full_wf.shape[1], rng
            )
            if plan is None:
                n_skip_plan += 1
                continue
            slice_start, new_p_list, new_s_list, samp_cat = plan
            slice_end = slice_start + T_SAMPLES

            front_pad_size = max(0, -slice_start)
            back_pad_size = max(0, slice_end - full_wf.shape[1])

            # Bug fix (ROMPLUS pad-size cap): legacy code allowed unbounded
            # front_pad / back_pad. For very short ROMPLUS SAC traces (e.g.
            # 100 samples), front_pad_size could reach 8900 → 99% synthetic
            # noise + 1% real signal labeled as a real earthquake event.
            # Model trains "mostly flat → pick" → FA on real flat noise at
            # inference. Cap each side at 2000 samples (20 s) AND total
            # synthetic at 2500 samples (~28%) — per-side bound prevents
            # extreme degenerate cases, combined bound prevents the corner
            # case where both sides are simultaneously near-max (1999 + 1999
            # = 44% synthetic, which the per-side check alone would allow).
            MAX_PAD_SAMPLES = 2000
            MAX_TOTAL_PAD_SAMPLES = 2500
            if (front_pad_size > MAX_PAD_SAMPLES or back_pad_size > MAX_PAD_SAMPLES
                    or (front_pad_size + back_pad_size) > MAX_TOTAL_PAD_SAMPLES):
                n_skip_pick += 1
                continue

            # Build the 9000-sample slice with optional spectrum-matched pad
            wf_out = np.zeros((N_CHANNELS, T_SAMPLES), dtype=np.float32)
            real_lo = max(0, slice_start)
            real_hi = min(full_wf.shape[1], slice_end)
            if real_hi <= real_lo:
                n_skip_pick += 1
                continue
            real_slice = full_wf[:, real_lo:real_hi]
            wf_out[:, front_pad_size:front_pad_size + real_slice.shape[1]] = real_slice

            # Spectrum-matched front pad referenced from pre-P samples
            if front_pad_size > 0:
                pre_p_end = min(tp_npts, full_wf.shape[1])
                for ch in range(N_CHANNELS):
                    pre = full_wf[ch, :pre_p_end] if pre_p_end >= 200 else full_wf[ch, :500]
                    ref = find_reference_signal(
                        pre, window_size=min(500, len(pre)),
                        max_search=len(pre), min_unique=50,
                    ) if len(pre) > 0 else np.zeros(500)
                    pad = generate_matching_noise(ref, front_pad_size).astype(np.float32)
                    wf_out[ch, :front_pad_size] = pad

            # Back pad spectrum-matched from post-S coda
            if back_pad_size > 0:
                tail_lo = min(full_wf.shape[1] - 1, ts_npts + 100)
                tail = full_wf[:, tail_lo:]
                for ch in range(N_CHANNELS):
                    coda = tail[ch] if tail.shape[1] >= 200 else full_wf[ch, -500:]
                    ref = find_reference_signal(
                        coda, window_size=min(500, len(coda)),
                        max_search=len(coda), min_unique=50,
                    ) if len(coda) > 0 else np.zeros(500)
                    pad = generate_matching_noise(ref, back_pad_size).astype(np.float32)
                    wf_out[ch, T_SAMPLES - back_pad_size:] = pad

            if not np.isfinite(wf_out).all():
                n_skip_align += 1
                continue

            # Final pick bounds (re-check after padding builds the window).
            if any(not (0 <= int(p) < T_SAMPLES) for p in new_p_list):
                n_skip_pick += 1
                continue
            if any(not (0 <= int(s) < T_SAMPLES) for s in new_s_list):
                n_skip_pick += 1
                continue

            # Optional subclass mutation (zeropad does channel drop).
            post = self._post_window(wf_out, new_p_list, new_s_list, rng)
            if post is None:
                continue
            wf_final, new_p_list, new_s_list, category = post

            # Stratification: only applies when samp_cat matches a bin key.
            # Ponly / Sonly samp_cats sit outside the singleEQ bin set so
            # they pass through freely even if stratify_ps_bins is True.
            if bin_caps and samp_cat in bin_caps:
                if bin_counts[samp_cat] >= bin_caps[samp_cat]:
                    n_skip_bin_full += 1
                    continue

            sample_id = f"{self.category}_{year}_{event}_{net}.{sta}.{loc}.{channel}"
            yield WaveformSample(
                sample_id=sample_id,
                waveform=wf_final,
                category=category,
                sampling_category=samp_cat,
                split=split,
                source_file=str(self.sac_root),
                p_arrival_samples=[int(p) for p in new_p_list],
                s_arrival_samples=[int(s) for s in new_s_list],
                polarity="",
            )
            if bin_caps and samp_cat in bin_caps:
                bin_counts[samp_cat] += 1
            n_yielded += 1

        log.info(
            "ROMPLUS%sAdapter: yielded=%d  skip_sac=%d  skip_align=%d  "
            "skip_pick=%d  skip_plan=%d  skip_bin_full=%d",
            self.category, n_yielded, n_skip_sac, n_skip_align, n_skip_pick,
            n_skip_plan, n_skip_bin_full,
        )
        if bin_caps:
            log.info("ROMPLUS%s per-bin yield: %s",
                     self.category, dict(bin_counts))


class ROMPLUSSingleEQAdapter(_ROMPLUSBaseAdapter):
    name = "ROMPLUS"
    category = "singleEQ"
    split_salt = "ROMPLUS-singleEQ-v1"


class ROMPLUSSingleEQZeropadAdapter(_ROMPLUSBaseAdapter):
    """singleEQ variant with random horizontal channel-drop augmentation."""
    name = "ROMPLUS"
    category = "singleEQ_zeropad"
    split_salt = "ROMPLUS-singleEQ_zeropad-v1"

    def __init__(
        self,
        sac_root: Path,
        metadata_csv: Path,
        *,
        drop_prob: float = 0.5,
        **kwargs,
    ):
        # Honour the explicit ``split_salt`` class attr; the base ``__init__``
        # only overrides it when the caller passes one explicitly.
        super().__init__(sac_root, metadata_csv, **kwargs)
        self.drop_prob = drop_prob

    def _post_window(self, wf, p_list, s_list, rng):
        wf = wf.copy()
        if rng.random() < self.drop_prob:
            # Zero one of the horizontal channels (never Z, the dominant signal)
            if rng.random() < 0.5:
                wf[0] = 0.0
            else:
                wf[1] = 0.0
        return wf, p_list, s_list, self.category


class ROMPLUSPonlyAdapter(_ROMPLUSBaseAdapter):
    """ROMPLUS P-only: window contains P, S deliberately falls outside.

    ROMPLUS metadata always has both ``tp_utc`` and ``ts_utc``, so we
    deliberately push the window left enough that the S arrival sits past the
    9 000-sample window's end (with a 200-sample exclusion margin).

    Math (analogous to :class:`CREWPonlyAdapter`):

        slice_end  = (tp_npts - target_p) + T_SAMPLES
        slice_end  ≤ ts_npts - PHASE_EXCLUSION_MARGIN
        ⇒ target_p ≥ T_SAMPLES + PHASE_EXCLUSION_MARGIN - ps_residual

    Combined with ``min_p_position`` / ``max_p_position`` knobs:

        target_p ∈ [max(min_p_position, T_SAMPLES + margin - ps_residual),
                    max_p_position]

    Rows where the lower bound exceeds ``max_p_position`` are skipped (P-S gap
    too tight to fit a 90 s window after pushing S past the end). Default
    ``max_p_position = 8500`` is intentionally larger than the singleEQ default
    (7000) to capture small-P-S-gap rows, with the spectrum-matched front pad
    handling negative ``slice_start`` values.
    """
    name = "ROMPLUS"
    category = "Ponly"
    split_salt = "ROMPLUS-Ponly-v1"

    def __init__(
        self,
        sac_root: Path,
        metadata_csv: Path,
        *,
        min_p_position: int = 1000,
        max_p_position: int = 8500,
        **kwargs,
    ):
        super().__init__(
            sac_root, metadata_csv,
            min_p_position=min_p_position,
            max_p_position=max_p_position,
            **kwargs,
        )

    def _compute_slice_plan(self, tp_npts, ts_npts, full_npts, rng):
        ps_residual = ts_npts - tp_npts
        if ps_residual <= 0:
            return None

        lower = self.min_p_position
        upper = self.max_p_position

        # Push window left so S falls past T_SAMPLES with the margin.
        min_p_for_s_excl = T_SAMPLES + self.PHASE_EXCLUSION_MARGIN - ps_residual
        lower = max(lower, min_p_for_s_excl)

        if lower > upper:
            return None

        target_p = int(rng.integers(lower, upper + 1))
        slice_start = tp_npts - target_p

        # Final sanity: S must not fall inside the window.
        s_in_win = ts_npts - slice_start
        if 0 <= s_in_win < T_SAMPLES:
            return None

        new_p = target_p
        if not (0 <= new_p < T_SAMPLES):
            return None
        return slice_start, [new_p], [], self.category


class ROMPLUSSonlyAdapter(_ROMPLUSBaseAdapter):
    """ROMPLUS S-only: window contains S, P deliberately falls outside.

    Math (analogous to :class:`CREWSonlyAdapter`):

        slice_start = ts_npts - target_s
        slice_start ≥ tp_npts + PHASE_EXCLUSION_MARGIN
        ⇒ target_s ≤ ps_residual - PHASE_EXCLUSION_MARGIN

    Combined with ``min_s_position`` / ``max_s_position`` knobs:

        target_s ∈ [min_s_position,
                    min(max_s_position, ps_residual - margin)]

    Rows with ``ps_residual ≤ min_s_position + margin`` (i.e. P-S gap too
    short for the chosen min_s_position) are skipped. Default
    ``min_s_position = 500`` keeps the captured row count high since most
    ROMPLUS events have ps_residual ≥ 700.
    """
    name = "ROMPLUS"
    category = "Sonly"
    split_salt = "ROMPLUS-Sonly-v1"

    def __init__(
        self,
        sac_root: Path,
        metadata_csv: Path,
        *,
        min_s_position: int = 500,
        max_s_position: int = 8000,
        **kwargs,
    ):
        super().__init__(sac_root, metadata_csv, **kwargs)
        self.min_s_position = min_s_position
        self.max_s_position = max_s_position

    def _compute_slice_plan(self, tp_npts, ts_npts, full_npts, rng):
        ps_residual = ts_npts - tp_npts
        if ps_residual <= 0:
            return None

        lower = self.min_s_position
        upper = self.max_s_position

        # Push window right so P falls below 0 with the margin.
        max_s_for_p_excl = ps_residual - self.PHASE_EXCLUSION_MARGIN
        upper = min(upper, max_s_for_p_excl)

        if lower > upper:
            return None

        target_s = int(rng.integers(lower, upper + 1))
        slice_start = ts_npts - target_s

        # Final sanity: P must not fall inside the window.
        p_in_win = tp_npts - slice_start
        if 0 <= p_in_win < T_SAMPLES:
            return None

        new_s = target_s
        if not (0 <= new_s < T_SAMPLES):
            return None
        return slice_start, [], [new_s], self.category
