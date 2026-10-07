"""CEED_NC singleEQ adapter reading from a pre-built 90 s redpan H5.

Source layout (see builder/README.md, CEED):

    <ceed_nc_h5_dir>/
        CEED_NC_dataset_90s_singleEQ.h5            # waveforms (already 9000-sample)
        CEED_NC_dataset_90s_singleEQ_metadata.csv  # rows align row-by-row to H5
        CEED_NC_dataset_90s_singleEQ_polarity_metadata.csv  # same H5, +polarity

The H5 stores one resizable 3D dataset per (category, split):
    /singleEQ/{train,val,test}/waveforms : (N, 3, 9000) float32

Because the source is already 90 s @ 100 Hz channel-first (E, N, Z), the
adapter does NOT re-pad or re-slice — it just repacks into the unified
WaveformSample schema. Optionally enriches with polarity from the parallel
metadata CSV when ``with_polarity=True``.

Split policy
------------
This adapter is an H5 *repack*: it preserves the source's native
``train`` / ``val`` / ``test`` group structure verbatim. It does NOT call
``bernoulli_3way`` or ``hash_split`` from ``builder.splits`` —
the source H5 already carries a 3-way split that the audit confirmed is
byte-equal-correct, and re-splitting would discard that work and risk
event leakage.

Polarity dictionary in source CSV: 'U' (up), 'D' (down), 'N' (analyst
examined the pick, no clear first motion = "emergent"), plus NaN (no
annotation). Mapped onto the unified ``{'U', 'D', 'N', ''}`` dictionary:
'U'→'U', 'D'→'D', 'N'→'N' (kept distinct so the impulsive head / [N,U,D]
softmax head get real emergent supervision), NaN/anything-else→''.
"""
from __future__ import annotations
import ast
import logging
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np
import pandas as pd

from .base import BaseAdapter
from ..schema import (
    WaveformSample, T_SAMPLES, N_CHANNELS, SAMPLING_RATE_HZ,
)


log = logging.getLogger(__name__)


def _ps_diff_bin(ps_diff_samples: int) -> str:
    """Match train_*.json category_weights bin keys."""
    sec = ps_diff_samples / SAMPLING_RATE_HZ
    if sec < 5:   return "singleEQ_00-05s"
    if sec < 10:  return "singleEQ_05-10s"
    if sec < 15:  return "singleEQ_10-15s"
    if sec < 20:  return "singleEQ_15-20s"
    return "singleEQ_20s_plus"


def _parse_picks(s) -> list[int]:
    """Parse a CSV pick cell. Source format is e.g. '[4542]' or '[]'.

    Uses ``int(round(...))`` rather than bare ``int(...)`` so fractional
    sample indices (which round-trip through float in pandas) don't bias
    picks toward zero by truncation.
    """
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s or s == "[]":
        return []
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    if isinstance(v, (list, tuple)):
        return [int(round(float(x))) for x in v]
    return [int(round(float(v)))]


def _polarity_from_str(s) -> str:
    """Map CEED_NC's p_polarity (U/D/N) into the unified {'U','D','N',''} dictionary.

    'N' (analyst examined the pick, no clear first motion) is kept DISTINCT
    from '' (no annotation) — see module docstring. Anything not in {U,D,N},
    incl. NaN, → ''.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip().upper()
    if s == "U":
        return "U"
    if s == "D":
        return "D"
    if s == "N":
        return "N"
    return ""   # unknown / NaN / unannotated


_VALID_SPLITS = {"train", "val", "test"}


class CEEDNCSingleEQAdapter(BaseAdapter):
    """CEED_NC single-event: stream samples directly from the pre-built H5.

    The source H5 already holds 9000-sample channel-first traces, so this
    adapter is a faithful repack: read row -> (3, 9000) -> WaveformSample.

    Args:
        h5_path     : path to ``CEED_NC_dataset_90s_singleEQ.h5``.
        csv_path    : metadata CSV. If ``with_polarity`` is True the caller
                      should pass the ``*_singleEQ_metadata.csv`` (it carries
                      the ``p_polarity`` column). Either CSV is row-aligned
                      with the H5 via ``hdf5_index``+``split``.
        with_polarity: include first-motion labels (U/D, '' for N/unknown).

    Splits are taken verbatim from the source CSV ('train' | 'val' | 'test').
    Rows whose ``split`` is missing or outside that set are skipped with a
    debug log; the adapter does NOT invent splits via ``bernoulli_3way`` or
    ``hash_split`` — the source H5 already provides a vetted 3-way partition.
    """
    name = "CEED_NC"
    category = "singleEQ"

    def __init__(
        self,
        h5_path: Path,
        csv_path: Path,
        *,
        with_polarity: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.h5_path = Path(h5_path)
        self.csv_path = Path(csv_path)
        self.with_polarity = with_polarity

    def __iter__(self) -> Iterator[WaveformSample]:
        log.info("CEEDNCSingleEQAdapter: reading %s", self.csv_path)
        df = pd.read_csv(self.csv_path, low_memory=False)
        log.info("  rows in metadata: %d", len(df))

        # Schema sanity
        required_cols = {"hdf5_index", "sample_id", "split",
                         "p_arrival_sample", "s_arrival_sample"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"missing CSV columns: {missing}")
        if self.with_polarity and "p_polarity" not in df.columns:
            raise ValueError(
                "with_polarity=True requires a CSV with 'p_polarity' column "
                f"(got columns: {list(df.columns)})"
            )

        n_yielded = 0
        with h5py.File(self.h5_path, "r") as hf:
            # Cache dataset handles per split for fast row reads.
            ds_per_split: dict[str, h5py.Dataset] = {}

            for _, row in df.iterrows():
                if self.max_samples is not None and n_yielded >= self.max_samples:
                    break

                raw_split = row.get("split")
                if pd.isna(raw_split):
                    log.debug("  row %s: missing split; skip",
                              row.get("sample_id"))
                    continue
                split = str(raw_split).strip()
                if split not in _VALID_SPLITS:
                    log.debug("  row %s: unrecognized split %r; skip",
                              row.get("sample_id"), split)
                    continue
                if self.split_filter and split != self.split_filter:
                    continue

                hidx = int(row["hdf5_index"])
                sid = str(row["sample_id"])

                # Lazy dataset open per split
                if split not in ds_per_split:
                    ds_path = f"singleEQ/{split}/waveforms"
                    if ds_path not in hf:
                        log.warning("  no dataset %s in H5; skipping split %s",
                                    ds_path, split)
                        ds_per_split[split] = None  # type: ignore[assignment]
                        continue
                    ds_per_split[split] = hf[ds_path]
                ds = ds_per_split[split]
                if ds is None:
                    continue

                if hidx < 0 or hidx >= ds.shape[0]:
                    log.warning("  hdf5_index %d out of range for %s/%s",
                                hidx, split, sid)
                    continue

                wf = np.asarray(ds[hidx], dtype=np.float32)
                if wf.shape != (N_CHANNELS, T_SAMPLES):
                    log.warning("  unexpected shape %s for %s; skip",
                                wf.shape, sid)
                    continue
                if not np.isfinite(wf).all():
                    log.warning("  non-finite waveform for %s; skip", sid)
                    continue

                p_picks = _parse_picks(row["p_arrival_sample"])
                s_picks = _parse_picks(row["s_arrival_sample"])
                if not p_picks or not s_picks:
                    continue
                # singleEQ contract requires single P/S pair
                p_picks = p_picks[:1]
                s_picks = s_picks[:1]
                if not (0 <= p_picks[0] < T_SAMPLES and 0 <= s_picks[0] < T_SAMPLES):
                    continue
                ps_residual = s_picks[0] - p_picks[0]
                if ps_residual <= 0:
                    continue

                pol = ""
                if self.with_polarity:
                    pol = _polarity_from_str(row.get("p_polarity"))

                # Prefer source-provided sampling_category bin; fall back to recompute
                bin_str = row.get("sampling_category")
                if not isinstance(bin_str, str) or not bin_str:
                    bin_str = _ps_diff_bin(ps_residual)

                yield WaveformSample(
                    sample_id=f"singleEQ_{sid}" if not sid.startswith("singleEQ_") else sid,
                    waveform=wf.copy(),
                    category="singleEQ",
                    sampling_category=bin_str,
                    split=split,  # type: ignore[arg-type]
                    source_file=str(self.h5_path),
                    p_arrival_samples=[int(round(float(p_picks[0])))],
                    s_arrival_samples=[int(round(float(s_picks[0])))],
                    polarity=pol,  # type: ignore[arg-type]
                )
                n_yielded += 1
