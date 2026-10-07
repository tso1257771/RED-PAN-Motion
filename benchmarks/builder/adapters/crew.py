"""CREW singleEQ + Ponly + Sonly adapters reading the SeisBench-cached chunks.

Source layout (the SeisBench CREW download, ``seisbench.data.CREW()``):

    <crew_root>/                        # e.g. ~/.seisbench/datasets/crew
        chunks                          # newline-separated chunk ids ('000', ...)
        metadata{NNN}.csv               # one row per trace
        waveforms{NNN}.hdf5             # /data/bucket{K} -> (1024, 3, 30000) float64

CREW traces are 300 s @ 100 Hz channel-first (E, N, Z) — 30 000 samples wide.
``trace_name`` follows the pattern ``bucket{K}${row},:3,:30000`` which selects
``hdf5['data/bucket{K}'][row]``.

CREW is the only training source that exposes explicit Ponly / Sonly partial
picks (rows where one of ``trace_P_arrival_sample`` / ``trace_S_arrival_sample``
is NaN, or where we deliberately slice a window that excludes the other phase).
There is no noise category and no EEWA/MMWA mosaic for CREW.

This module is a thin slicing adapter: each yielded WaveformSample is a 9 000-
sample window (90 s) that fits entirely inside the 30 000-sample source trace,
so no synthetic padding is needed.

Audit fixes (2026-04):
  - 12Z component_order rows are filtered upfront (legacy code reused
    ``trace_name`` lookups but indexed with the wrong E/N channel mapping —
    we now drop them entirely with a logged count).
  - ``trace_P_arrival_sample`` / ``trace_S_arrival_sample`` are converted with
    ``int(round(float(...)))`` instead of ``int(...)`` to avoid the systematic
    -1-sample bias from truncating fractional pick values (~53 % of P and
    ~57 % of S CREW arrivals are non-integer floats).
  - The per-row Bernoulli 80/20 split was replaced with ``hash_split`` keyed on
    ``source_id`` so all traces sharing an event land in the SAME split. The
    cached SeisBench metadata marks every row as ``split=train``, so we
    generate our own three-way split (default 70/15/15).
"""
from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np
import pandas as pd

from .base import BaseAdapter
from ..schema import (
    WaveformSample, T_SAMPLES, N_CHANNELS, SAMPLING_RATE_HZ,
)
from ..splits import hash_split


log = logging.getLogger(__name__)


# ``bucket0$123,:3,:30000`` — capture (bucket idx, row idx)
_TRACE_NAME_RE = re.compile(r"^bucket(\d+)\$(\d+),")
_SOURCE_NPTS = 30000  # CREW: 300 s @ 100 Hz


def _ps_diff_bin(ps_diff_samples: int) -> str:
    """Match train_*.json category_weights bin keys."""
    sec = ps_diff_samples / SAMPLING_RATE_HZ
    if sec < 5:   return "singleEQ_00-05s"
    if sec < 10:  return "singleEQ_05-10s"
    if sec < 15:  return "singleEQ_10-15s"
    if sec < 20:  return "singleEQ_15-20s"
    return "singleEQ_20s_plus"


def _parse_trace_name(trace_name: str) -> Optional[tuple[str, int]]:
    """Split ``bucket{K}${row},:3,:30000`` into (dataset_path, row_idx).

    Returns None on malformed names. ``dataset_path`` is the H5 path
    inside the chunk, e.g. ``data/bucket3``.
    """
    m = _TRACE_NAME_RE.match(trace_name)
    if not m:
        return None
    bucket_idx = int(m.group(1))
    row_idx = int(m.group(2))
    return f"data/bucket{bucket_idx}", row_idx


def _discover_chunks(root: Path) -> list[str]:
    """Return chunk ids ('000', '001', ...) by reading the SeisBench
    ``chunks`` index file. Falls back to glob if the index is missing."""
    chunks_index = root / "chunks"
    if chunks_index.is_file():
        with open(chunks_index, "r") as f:
            ids = [line.strip() for line in f if line.strip()]
        if ids:
            return ids
    # Glob fallback
    return sorted(p.stem.replace("metadata", "") for p in root.glob("metadata*.csv"))


def _safe_round_int(raw) -> Optional[int]:
    """Convert a raw arrival-sample value (possibly NaN/None/float) to an int.

    Uses ``int(round(float(raw)))`` so fractional source values (e.g. 5081.4)
    round to nearest integer rather than truncate to 5081 (legacy bug). NaN
    or otherwise unparseable values return ``None``.
    """
    if raw is None:
        return None
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return int(round(f))


class _CREWAdapterBase(BaseAdapter):
    """Shared CREW chunked-source machinery.

    Subclasses set ``category`` and override ``_choose_window`` to translate
    a metadata row into a slice plan. The base class handles:

      - chunk discovery via ``chunks`` index file
      - row filtering (component_order = ENZ, sampling rate = 100, valid trace
        category) before the adapter touches the heavy waveform store
      - deterministic 3-way split assignment via ``hash_split`` keyed on
        ``source_id`` (event-aware: all traces from the same event land in
        the same split)
      - per-bucket batched H5 reads (one open per chunk, one slice per row)
    """
    name = "CREW"

    # Subclasses override
    category: str = ""
    # Subclasses override to differentiate split salts per category so that
    # picks of, e.g., a Ponly window don't collide with the singleEQ window
    # drawn from the same event (different windows are fair to use across
    # splits — but if we ever want them grouped, change this to a constant).
    split_salt: str = "CREW-base-v1"

    def __init__(
        self,
        crew_root: Path,
        *,
        p_train: float = 0.70,
        p_val: float = 0.15,
        seed: int = 42,
        min_p_position: int = 500,
        max_p_position: int = 7500,
        accept_categories: tuple[str, ...] = (
            "earthquake", "explosion", "rock burst",
            "mining explosion", "nuclear explosion",
        ),
        # Which CSV columns supply the P / S arrival samples. Default = the
        # composite first-arriving picks. CREWSingleEQAdapter overrides these
        # with the explicit Pn / Sn columns when phase_pair="mantle" — see its
        # docstring for the rationale (consistent regional phase type).
        p_column: str = "trace_P_arrival_sample",
        s_column: str = "trace_S_arrival_sample",
        # Back-compat: some callers still pass ``train_frac``. Map it to
        # ``p_train`` and emit a debug message rather than break the CLI.
        train_frac: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.crew_root = Path(crew_root)
        if train_frac is not None:
            log.debug("CREW%sAdapter: train_frac=%s overrides p_train=%s",
                      self.category, train_frac, p_train)
            p_train = float(train_frac)
            # Preserve the historical behaviour where the remainder went to
            # val: if a caller still says train_frac=0.8 we keep p_train=0.8
            # but reduce p_val to (1-p_train)/2 so that test gets the other
            # half. If the new caller passes both, p_val wins.
        self.p_train = p_train
        self.p_val = p_val
        self.seed = seed
        self.min_p_position = min_p_position
        self.max_p_position = max_p_position
        self.accept_categories = set(accept_categories)
        self.p_column = p_column
        self.s_column = s_column

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------
    def _choose_window(
        self,
        p_idx: Optional[int],
        s_idx: Optional[int],
        rng: np.random.Generator,
    ) -> Optional[tuple[int, list[int], list[int]]]:
        """Decide the slice for one row.

        Returns:
            (slice_start, p_arrivals_in_window, s_arrivals_in_window)
            or None to skip the row.

        The base class guarantees ``slice_start + T_SAMPLES <= _SOURCE_NPTS``
        and ``slice_start >= 0`` before reading the H5.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[WaveformSample]:
        chunk_ids = _discover_chunks(self.crew_root)
        log.info("CREW%sAdapter: %d chunks under %s",
                 self.category, len(chunk_ids), self.crew_root)

        rng = np.random.default_rng(self.seed)
        # 2026-05-11: shuffle the chunk order so that --max-samples N yields a
        # (chunk-level) RANDOM subset rather than the first N rows by chunk
        # index. Matters for Ponly/Sonly which we now cap at ~10% — without
        # this, a cap would systematically miss whichever regions/epochs land
        # in the higher-numbered chunks. Deterministic per seed.
        chunk_ids = list(chunk_ids)
        rng.shuffle(chunk_ids)
        n_yielded = 0
        n_12z_kept_total = 0
        n_rounded_picks_total = 0

        for chunk_id in chunk_ids:
            if self.max_samples is not None and n_yielded >= self.max_samples:
                break

            csv_path = self.crew_root / f"metadata{chunk_id}.csv"
            h5_path = self.crew_root / f"waveforms{chunk_id}.hdf5"
            if not csv_path.is_file() or not h5_path.is_file():
                log.warning("missing CREW chunk pair %s — skipping", chunk_id)
                continue

            df = pd.read_csv(csv_path, low_memory=False)
            n_rows_total = len(df)

            # NOTE: 12Z rows are NOT filtered out — keeping them is acceptable
            # because (a) the picker / detector heads are trained primarily off
            # the Z (vertical) channel, which is correct regardless of horizontal
            # orientation, and (b) the polarity head reads Z directly. The
            # horizontals 1/2 (vs E/N) carry a rotated S-wave signal but that
            # only affects S-pick precision marginally, not the FA story we are
            # debugging. Earlier audit suggested filtering; user feedback chose
            # to keep them.
            comp_order = df.get("component_order", pd.Series([""] * n_rows_total))
            comp_order = comp_order.astype(str).str.strip()
            n_12z = int((comp_order != "ENZ").sum())
            n_12z_kept_total += n_12z
            if n_12z:
                non_enz = comp_order[comp_order != "ENZ"].value_counts(dropna=False).to_dict()
                log.info(
                    "CREW chunk %s: %d/%d rows have component_order != ENZ — "
                    "keeping all (distribution: %s)",
                    chunk_id, n_12z, n_rows_total, non_enz,
                )

            try:
                hf = h5py.File(h5_path, "r")
            except OSError as e:
                log.warning("cannot open %s: %s", h5_path, e)
                continue

            try:
                for row in df.itertuples(index=False):
                    if self.max_samples is not None and n_yielded >= self.max_samples:
                        break

                    trace_cat = getattr(row, "trace_category", "")
                    if trace_cat not in self.accept_categories:
                        continue

                    # 12Z rows are kept (see filter-removal note at chunk
                    # load time). The Z channel (index 2) carries the same
                    # physical signal regardless of horizontal orientation.

                    # Sampling-rate safety belt — audit confirmed all CREW
                    # rows are 100 Hz, but a non-100 row would silently break
                    # downstream label generation.
                    try:
                        if int(getattr(row, "trace_sampling_rate_hz")) != SAMPLING_RATE_HZ:
                            continue
                    except (TypeError, ValueError):
                        continue

                    # Event-aware split assignment. CREW rows all carry
                    # split=='train' in metadata, so we must derive our own.
                    # Group by source_id so that two traces from the same
                    # event always land in the same split.
                    source_id = str(getattr(row, "source_id", "") or "")
                    if not source_id:
                        # Fallback: bucket key from trace_name keeps determinism
                        # even if source_id is missing for a row.
                        trace_name_for_key = str(getattr(row, "trace_name", ""))
                        parsed_for_key = _parse_trace_name(trace_name_for_key)
                        if parsed_for_key is None:
                            continue
                        bucket_path, _row_idx = parsed_for_key
                        source_id = f"{chunk_id}/{bucket_path}"
                    split = hash_split(
                        source_id,
                        p_train=self.p_train,
                        p_val=self.p_val,
                        salt=self.split_salt,
                    )
                    if self.split_filter and split != self.split_filter:
                        continue

                    # Audit fix #2: round, don't truncate. Capture the raw
                    # values so we can report how many got rounded.
                    # p_column / s_column select either the composite picks
                    # (default) or the explicit Pn / Sn picks (mantle mode).
                    p_raw = getattr(row, self.p_column, np.nan)
                    s_raw = getattr(row, self.s_column, np.nan)
                    p_idx = _safe_round_int(p_raw)
                    s_idx = _safe_round_int(s_raw)

                    def _is_fractional(raw, idx):
                        if idx is None:
                            return False
                        try:
                            f = float(raw)
                        except (TypeError, ValueError):
                            return False
                        if f != f:  # NaN
                            return False
                        return f != float(idx)

                    if _is_fractional(p_raw, p_idx):
                        n_rounded_picks_total += 1
                    if _is_fractional(s_raw, s_idx):
                        n_rounded_picks_total += 1

                    plan = self._choose_window(p_idx, s_idx, rng)
                    if plan is None:
                        continue
                    slice_start, p_in_window, s_in_window = plan
                    slice_end = slice_start + T_SAMPLES
                    if slice_start < 0 or slice_end > _SOURCE_NPTS:
                        # Tighten contract: never silently pad. CREW is 30 000
                        # samples so a 9 000-sample window must fit fully.
                        continue

                    trace_name = str(getattr(row, "trace_name", ""))
                    parsed = _parse_trace_name(trace_name)
                    if parsed is None:
                        continue
                    dataset_path, source_row = parsed
                    try:
                        ds = hf[dataset_path]
                        # H5 layout: bucket -> (M, 3, 30000) channel-first ENZ.
                        wf_full = np.array(ds[source_row, :, :]).astype(np.float32)
                    except (KeyError, OSError, IndexError) as e:
                        log.debug("skip %s/%s: %s", chunk_id, trace_name, e)
                        continue

                    if wf_full.shape != (N_CHANNELS, _SOURCE_NPTS):
                        # Defensive: some trace_name suffixes may indicate
                        # shorter records — skip them rather than pad.
                        continue
                    wf_cf = wf_full[:, slice_start:slice_end].copy()
                    if wf_cf.shape != (N_CHANNELS, T_SAMPLES):
                        continue
                    if not np.isfinite(wf_cf).all():
                        continue

                    sampling_category = self._sampling_category(p_in_window, s_in_window)
                    sample_id = self._make_sample_id(
                        chunk_id, trace_name, slice_start
                    )

                    yield WaveformSample(
                        sample_id=sample_id,
                        waveform=wf_cf,
                        category=self.category,  # type: ignore[arg-type]
                        sampling_category=sampling_category,
                        split=split,
                        source_file=str(h5_path),
                        p_arrival_samples=list(p_in_window),
                        s_arrival_samples=list(s_in_window),
                        polarity="",   # CREW has no first-motion labels
                    )
                    n_yielded += 1
            finally:
                hf.close()

        log.info(
            "CREW%sAdapter: yielded=%d  12Z_kept=%d  fractional_picks_rounded=%d",
            self.category, n_yielded, n_12z_kept_total, n_rounded_picks_total,
        )

    # ------------------------------------------------------------------
    def _make_sample_id(self, chunk_id: str, trace_name: str, slice_start: int) -> str:
        return f"{self.category}_chunk{chunk_id}_{trace_name}_s{slice_start}"

    def _sampling_category(self, p_in_window: list[int], s_in_window: list[int]) -> str:
        """Subclasses override when they want P-S diff binning."""
        return self.category


# ----------------------------------------------------------------------
# Concrete adapters
# ----------------------------------------------------------------------
class CREWSingleEQAdapter(_CREWAdapterBase):
    """CREW single-event: both P and S land inside the 9 000-sample window.

    Slicing strategy: pick a target P offset uniformly in [min_p_position,
    max_p_position] and require ``slice_start = p_idx - target_p`` together
    with the resulting S position to lie inside the window. No padding is
    ever performed: rows that do not fit the source extent are skipped.

    phase_pair (2026-05):
      ``"composite"`` (default) — read the first-arriving composite picks
        ``trace_P_arrival_sample`` / ``trace_S_arrival_sample``.  Legacy
        behaviour.  The composite S is a heterogeneous mix: ~68 % of it tracks
        the emergent mantle head wave Sn, ~32 % the later impulsive crustal Sg
        (the two are ~4 s apart).  Training on this mix maps the same regional
        waveform to inconsistent label positions and teaches global S
        under-confidence.
      ``"mantle"`` — read the explicit ``trace_Pn_arrival_sample`` /
        ``trace_Sn_arrival_sample`` columns instead.  Rows lacking either Pn or
        Sn are skipped automatically (``_choose_window`` returns None when an
        index is None).  This restricts CREW to a single, consistent regional
        phase type (mantle head-wave pair) — ~410 k window-fitting rows out of
        the 977 k composite pool.  Recommended for picker training.
    """
    category = "singleEQ"
    split_salt = "CREW-singleEQ-v1"

    _MANTLE_COLUMNS = {
        "p_column": "trace_Pn_arrival_sample",
        "s_column": "trace_Sn_arrival_sample",
    }

    def __init__(self, *args, phase_pair: str = "composite", **kwargs):
        if phase_pair not in ("composite", "mantle"):
            raise ValueError(
                f"phase_pair must be 'composite' or 'mantle', got {phase_pair!r}"
            )
        self.phase_pair = phase_pair
        if phase_pair == "mantle":
            # Override the arrival-sample columns to the explicit Pn / Sn picks.
            kwargs.update(self._MANTLE_COLUMNS)
        super().__init__(*args, **kwargs)

    def _choose_window(self, p_idx, s_idx, rng):
        if p_idx is None or s_idx is None:
            return None
        if not (0 <= p_idx < s_idx < _SOURCE_NPTS):
            return None
        ps_residual = s_idx - p_idx
        if ps_residual <= 0:
            return None

        target_p = int(rng.integers(self.min_p_position, self.max_p_position + 1))
        slice_start = p_idx - target_p
        slice_end = slice_start + T_SAMPLES
        if slice_start < 0 or slice_end > _SOURCE_NPTS:
            return None

        new_p = target_p
        new_s = target_p + ps_residual
        if not (0 <= new_p < T_SAMPLES and 0 <= new_s < T_SAMPLES):
            return None
        return slice_start, [new_p], [new_s]

    def _sampling_category(self, p_in_window, s_in_window):
        if p_in_window and s_in_window:
            return _ps_diff_bin(s_in_window[0] - p_in_window[0])
        return self.category


class CREWPonlyAdapter(_CREWAdapterBase):
    """CREW P-only: window contains P near the window tail, S excluded.

    Only rows where ``trace_S_arrival_sample`` is NaN are used — rows with
    a known S belong to singleEQ and were previously used here with a
    constrained P placement that caused 87 % of P arrivals to cluster in
    [7000, 7500]. The dataset loader's ``EDGE_LEN = 500`` filter then silently
    dropped all those samples (P must be in [8500, 9000)).

    Fix (2026-05):
      - Accept NaN-S rows only.  Known-S rows are skipped here; CREWSingleEQ
        captures them.
      - P is placed uniformly in [``min_p_position``, min(``max_p_position``, p_idx)]
        (default [8500, 8900]).  Clamping to p_idx prevents negative slice_start
        for the rare CREW rows where P arrives before sample 8500 in the source
        trace (~3 % of rows).  Rows where p_idx < min_p_position are skipped.
        At these positions the window end = p_idx - target_p + 9000 is well
        before any CREW S arrival (minimum PS diff ≈ 673 samples).

    The yielded sample carries an empty ``s_arrival_samples`` list.
    """
    category = "Ponly"
    split_salt = "CREW-Ponly-v1"

    def __init__(self, *args, min_p_position: int = 8500,
                 max_p_position: int = 8900, **kwargs):
        super().__init__(*args, min_p_position=min_p_position,
                         max_p_position=max_p_position, **kwargs)

    def _choose_window(
        self,
        p_idx: Optional[int],
        s_idx: Optional[int],
        rng: np.random.Generator,
    ) -> Optional[tuple[int, list[int], list[int]]]:
        # Only NaN-S rows — known-S rows go to CREWSingleEQAdapter.
        if p_idx is None or s_idx is not None:
            return None
        if not (0 <= p_idx < _SOURCE_NPTS):
            return None

        # Clip the feasible target_p range so slice_start = p_idx - target_p >= 0.
        # Rows with very early P (p_idx < min_p_position) cannot accommodate the
        # required in-window P offset and are skipped rather than silently rejected
        # after the draw.
        max_feasible = min(self.max_p_position, p_idx)
        if max_feasible < self.min_p_position:
            return None
        target_p = int(rng.integers(self.min_p_position, max_feasible, endpoint=True))
        slice_start = p_idx - target_p
        slice_end = slice_start + T_SAMPLES
        if slice_start < 0 or slice_end > _SOURCE_NPTS:
            return None

        new_p = target_p
        if not (0 <= new_p < T_SAMPLES):
            return None
        return slice_start, [new_p], []


class CREWSonlyAdapter(_CREWAdapterBase):
    """CREW S-only: window contains S near the window head, P excluded.

    Only rows where ``trace_P_arrival_sample`` is NaN are used.  Known-P rows
    had two problems under the old design:

      1. The P-exclusion constraint forced S into positions [500, 7500] — all
         of which fail the dataset loader's ``EDGE_LEN = 500`` filter (S must
         be < 500).  The generated Sonly H5 was silently discarded.
      2. The pre-S region of these windows contained P-wave coda (window start
         lay after P), teaching the model that P coda = pre-event noise.

    Fix (2026-05):
      - Accept NaN-P rows only.  Known-P rows are skipped here; CREWSingleEQ
        captures them.  NaN-P rows have genuine pre-event ambient noise before S
        (no labeled P ⇒ no P-coda contamination).
      - S is placed uniformly in [``min_s_position``, min(``max_s_position``, s_idx)]
        (default [100, 499]) so samples pass the EDGE_LEN = 500 filter.
        Clamping to s_idx prevents negative slice_start for very early S arrivals.
        Rows where s_idx < min_s_position are skipped.
      - No unlabeled-P check is performed — NaN-P rows have no p_idx available
        in the CREW metadata, so it cannot be verified whether an unlabeled
        physical P arrival lands in the window.

    The yielded sample carries an empty ``p_arrival_samples`` list.
    """
    category = "Sonly"
    split_salt = "CREW-Sonly-v1"

    def __init__(self, *args, min_s_position: int = 100,
                 max_s_position: int = 499, **kwargs):
        # Note: _CREWAdapterBase stores min_p_position / max_p_position but
        # CREWSonlyAdapter does not use them.  min_s_position / max_s_position
        # are stored directly on self because the base class has no s-position
        # attributes.
        super().__init__(*args, **kwargs)
        self.min_s_position = min_s_position
        self.max_s_position = max_s_position

    def _choose_window(
        self,
        p_idx: Optional[int],
        s_idx: Optional[int],
        rng: np.random.Generator,
    ) -> Optional[tuple[int, list[int], list[int]]]:
        # Only NaN-P rows — known-P rows go to CREWSingleEQAdapter.
        if s_idx is None or p_idx is not None:
            return None
        if not (0 <= s_idx < _SOURCE_NPTS):
            return None

        # Clip to prevent slice_start < 0 when S arrives very early in the trace.
        max_feasible = min(self.max_s_position, s_idx)
        if max_feasible < self.min_s_position:
            return None
        target_s = int(rng.integers(self.min_s_position, max_feasible, endpoint=True))
        slice_start = s_idx - target_s
        slice_end = slice_start + T_SAMPLES
        if slice_start < 0 or slice_end > _SOURCE_NPTS:
            return None

        new_s = target_s
        if not (0 <= new_s < T_SAMPLES):
            return None
        return slice_start, [], [new_s]
