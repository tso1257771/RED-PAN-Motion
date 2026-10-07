"""CEED_SC singleEQ adapter — sharded pre-built 90 s redpan H5s.

Source layout (Southern California subset of CEED):

    <ceed_sc_h5_dir>/
        CEED_SC_singleEQ_s000.h5
        CEED_SC_singleEQ_s000_metadata.csv
        CEED_SC_singleEQ_s000_metadata_polarity.csv
        ...
        CEED_SC_singleEQ_s038.h5
        CEED_SC_singleEQ_s038_metadata.csv
        CEED_SC_singleEQ_s038_metadata_polarity.csv

Each shard's H5 stores one resizable 3D dataset for the shard's split:
    /singleEQ/{train|val|test}/waveforms : (N, 3, 9000) float32

Each shard contains exactly one split (train, val, or test) — splits are
pre-baked into the shard layer, not multiplexed within a shard. The metadata
CSV's ``hdf5_index`` column references rows inside that shard's dataset.

Two metadata variants ship per shard:

* ``*_metadata.csv``           — string ``p_polarity`` column ('U', 'D', 'N').
* ``*_metadata_polarity.csv``  — integer ``polarity`` list-string per row,
                                  e.g. ``[1]`` (=U), ``[-1]`` (=D), ``[0]``
                                  (=unknown), with multi-event rows like
                                  ``[1, -1]``. Aligned per-pick to the
                                  ``p_arrival_sample`` list.

Because the source H5 is already 9000-sample channel-first (E, N, Z) at
100 Hz, this adapter does NOT re-pad or re-slice — it just repacks into the
unified ``WaveformSample`` schema. Optionally enriches with polarity from the
polarity-CSV when ``with_polarity=True``.

Split policy
------------
Like ``CEEDNCSingleEQAdapter``, this is an H5 *repack*: each shard's
``train``/``val``/``test`` group placement is preserved verbatim. The adapter
does NOT call ``bernoulli_3way`` or ``hash_split`` from
``builder.splits`` — the upstream sharding already provides a
3-way partition that the audit confirmed is byte-equal-correct. Rows whose
``split`` cell is missing or outside ``{'train','val','test'}`` are skipped
with a debug log.

Polarity dictionary in unified schema is ``{'U', 'D', 'N', ''}``. Source 'N'
(string CSV) and ``0`` (int-list CSV) — analyst examined the pick, no clear
first motion = "emergent" — both map to ``'N'`` (kept distinct from ``''`` so
the impulsive head / [N,U,D] softmax head get real emergent supervision). A
pick with no annotation at all (empty cell / NaN) → ``''``. Multi-pick rows
are flattened to the first P/S pair (matching CEED_NC's contract).
"""
from __future__ import annotations
import ast
import logging
from pathlib import Path
from typing import Iterator, List, Optional

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


def _parse_picks(s) -> List[int]:
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
    """Map ``p_polarity`` (U/D/N) into the unified ``{'U','D','N',''}`` dictionary.

    'N' (analyst examined, no clear first motion) is kept distinct from ''
    (no annotation). Anything not in {U,D,N}, incl. NaN, → ''.
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


def _polarity_from_int_list(s) -> str:
    """Map the integer ``polarity`` list-cell of the polarity-CSV onto the
    unified dictionary. Encoding observed in CEED_SC:

        1 → 'U'        -1 → 'D'        0 → 'N'   (examined, no clear first motion)

    A pick that's absent from the list (empty cell / NaN / unparseable) → ''
    (no annotation). Multi-pick rows expose a list per pick; the singleEQ
    contract retains only the first pair, so we read the first element.
    """
    picks = _parse_picks(s)
    if not picks:
        return ""
    first = picks[0]
    if first == 1:
        return "U"
    if first == -1:
        return "D"
    if first == 0:
        return "N"
    return ""   # unexpected encoding → treat as unannotated


_VALID_SPLITS = {"train", "val", "test"}


def _discover_shards(root: Path) -> List[Path]:
    """Find all ``CEED_SC_singleEQ_s???.h5`` shards under ``root``.

    Sorting is lexicographic on the file stem so that shards are processed
    in their numerical order (s000, s001, ..., s038) — this gives stable,
    reproducible row ordering in the output H5.
    """
    return sorted(root.glob("CEED_SC_singleEQ_s*.h5"))


def _csv_for_shard(h5_path: Path, with_polarity: bool) -> Path:
    """Pick the metadata CSV partner for ``h5_path``.

    When ``with_polarity=True`` we prefer ``*_metadata_polarity.csv`` because
    that file carries the integer ``polarity`` list aligned to picks. If the
    polarity CSV is missing for that shard we fall back to the plain
    ``*_metadata.csv`` (which still has the U/D/N ``p_polarity`` column on
    CEED_SC, so polarity is not lost — just first-pick only).

    When ``with_polarity=False`` we always use ``*_metadata.csv``.
    """
    base = h5_path.with_suffix("")  # strip .h5
    plain = base.parent / f"{base.name}_metadata.csv"
    polar = base.parent / f"{base.name}_metadata_polarity.csv"
    if with_polarity and polar.exists():
        return polar
    return plain


class CEEDSCSingleEQAdapter(BaseAdapter):
    """CEED_SC single-event: stream samples from sharded pre-built H5s.

    The source H5 already holds 9000-sample channel-first traces, so this
    adapter is a faithful repack: read row -> (3, 9000) -> WaveformSample.
    Shards are concatenated in lexical order across all splits.

    Args:
        root          : directory containing ``CEED_SC_singleEQ_s???.h5``.
        with_polarity : include first-motion labels (U/D, '' for N/unknown).
                        When True we prefer the ``*_metadata_polarity.csv``
                        partner per shard; else the plain ``*_metadata.csv``.

    Splits are taken verbatim from each shard's CSV. Each shard is a
    single-split file in CEED_SC, so the resulting H5 still has the
    standard ``/singleEQ/{train|val|test}/waveforms`` hierarchy. Rows with
    a missing or unrecognized ``split`` cell (anything outside
    ``{'train','val','test'}``) are skipped with a debug log; the adapter
    does NOT invent splits via ``bernoulli_3way`` or ``hash_split``.
    """
    name = "CEED_SC"
    category = "singleEQ"

    def __init__(
        self,
        *,
        root: Path,
        with_polarity: bool = False,
        max_samples: Optional[int] = None,
        split_filter: Optional[str] = None,
    ):
        super().__init__(max_samples=max_samples, split_filter=split_filter)
        self.root = Path(root)
        self.with_polarity = with_polarity

    def _required_cols(self, with_polarity: bool, df_cols: set) -> set:
        base = {"hdf5_index", "sample_id", "split",
                "p_arrival_sample", "s_arrival_sample"}
        if with_polarity:
            # Either schema is acceptable: polarity CSV uses 'polarity',
            # plain CSV uses 'p_polarity'. We just need ONE polarity column.
            if "polarity" not in df_cols and "p_polarity" not in df_cols:
                base.add("polarity")  # force failure with informative message
        return base

    def __iter__(self) -> Iterator[WaveformSample]:
        shards = _discover_shards(self.root)
        if not shards:
            raise FileNotFoundError(
                f"no CEED_SC shards found under {self.root!r}"
            )
        log.info("CEEDSCSingleEQAdapter: discovered %d shards under %s",
                 len(shards), self.root)

        n_yielded = 0
        for h5_path in shards:
            if self.max_samples is not None and n_yielded >= self.max_samples:
                break

            csv_path = _csv_for_shard(h5_path, self.with_polarity)
            if not csv_path.exists():
                log.warning("  no metadata CSV for shard %s; skipping",
                            h5_path.name)
                continue

            df = pd.read_csv(csv_path, low_memory=False)
            df_cols = set(df.columns)
            required = self._required_cols(self.with_polarity, df_cols)
            missing = required - df_cols
            if missing:
                raise ValueError(
                    f"shard {h5_path.name}: CSV missing columns {missing} "
                    f"(have: {sorted(df_cols)})"
                )

            # Decide which polarity column (if any) we'll read for this shard.
            polarity_col: Optional[str] = None
            polarity_is_int_list = False
            if self.with_polarity:
                if "polarity" in df_cols:
                    polarity_col = "polarity"
                    polarity_is_int_list = True
                elif "p_polarity" in df_cols:
                    polarity_col = "p_polarity"
                    polarity_is_int_list = False

            log.info("  shard %s: %d rows, csv=%s, polarity_col=%s",
                     h5_path.name, len(df), csv_path.name, polarity_col)

            with h5py.File(h5_path, "r") as hf:
                # Shard-scoped dataset cache (keyed by split). Each shard
                # actually holds one split, but we keep this defensive.
                ds_per_split: dict[str, Optional[h5py.Dataset]] = {}

                for _, row in df.iterrows():
                    if (
                        self.max_samples is not None
                        and n_yielded >= self.max_samples
                    ):
                        break

                    raw_split = row.get("split")
                    if pd.isna(raw_split):
                        log.debug(
                            "    shard %s: row %s missing split; skip",
                            h5_path.name, row.get("sample_id"),
                        )
                        continue
                    split = str(raw_split).strip()
                    if split not in _VALID_SPLITS:
                        log.debug(
                            "    shard %s: row %s unrecognized split %r; skip",
                            h5_path.name, row.get("sample_id"), split,
                        )
                        continue
                    if self.split_filter and split != self.split_filter:
                        continue

                    sid = str(row["sample_id"])
                    try:
                        hidx = int(row["hdf5_index"])
                    except (TypeError, ValueError):
                        continue

                    if split not in ds_per_split:
                        ds_path = f"singleEQ/{split}/waveforms"
                        if ds_path not in hf:
                            log.warning(
                                "    shard %s: no dataset %s; skipping split %s",
                                h5_path.name, ds_path, split,
                            )
                            ds_per_split[split] = None
                            continue
                        ds_per_split[split] = hf[ds_path]
                    ds = ds_per_split[split]
                    if ds is None:
                        continue

                    if hidx < 0 or hidx >= ds.shape[0]:
                        log.warning(
                            "    shard %s: hdf5_index %d out of range for %s/%s",
                            h5_path.name, hidx, split, sid,
                        )
                        continue

                    wf = np.asarray(ds[hidx], dtype=np.float32)
                    if wf.shape != (N_CHANNELS, T_SAMPLES):
                        log.warning("    shard %s: unexpected shape %s for %s; skip",
                                    h5_path.name, wf.shape, sid)
                        continue
                    if not np.isfinite(wf).all():
                        log.warning("    shard %s: non-finite waveform for %s; skip",
                                    h5_path.name, sid)
                        continue

                    p_picks = _parse_picks(row["p_arrival_sample"])
                    s_picks = _parse_picks(row["s_arrival_sample"])
                    if not p_picks or not s_picks:
                        continue
                    # singleEQ contract requires single P/S pair
                    p_picks = p_picks[:1]
                    s_picks = s_picks[:1]
                    if not (
                        0 <= p_picks[0] < T_SAMPLES
                        and 0 <= s_picks[0] < T_SAMPLES
                    ):
                        continue
                    ps_residual = s_picks[0] - p_picks[0]
                    if ps_residual <= 0:
                        continue

                    pol = ""
                    if self.with_polarity and polarity_col is not None:
                        cell = row[polarity_col]
                        pol = (
                            _polarity_from_int_list(cell)
                            if polarity_is_int_list
                            else _polarity_from_str(cell)
                        )

                    # Prefer source-provided sampling_category bin; fall back
                    # to recompute from ps_residual.
                    bin_str = row.get("sampling_category")
                    if not isinstance(bin_str, str) or not bin_str:
                        bin_str = _ps_diff_bin(ps_residual)

                    yield WaveformSample(
                        sample_id=(
                            sid if sid.startswith("singleEQ_")
                            else f"singleEQ_{sid}"
                        ),
                        waveform=wf.copy(),
                        category="singleEQ",
                        sampling_category=bin_str,
                        split=split,  # type: ignore[arg-type]
                        source_file=str(h5_path),
                        p_arrival_samples=[int(round(float(p_picks[0])))],
                        s_arrival_samples=[int(round(float(s_picks[0])))],
                        polarity=pol,  # type: ignore[arg-type]
                    )
                    n_yielded += 1
