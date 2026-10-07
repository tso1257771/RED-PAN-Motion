"""H5 + metadata-CSV writer.

Streams WaveformSample objects from a BaseAdapter into a resizable HDF5 file
with the standard hierarchy:

    /{category}/{split}/waveforms : (N, 3, 9000) float32

and an append-mode CSV index with these columns:

    hdf5_index, sample_id, category, sampling_category, split,
    source_file, p_arrival_sample, s_arrival_sample, ps_diff_samples,
    polarity, content_hash

The writer is intentionally thin: validation is the dataclass's job, decisions
about windowing/padding/filtering are the adapter's job.
"""
from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from typing import Optional, Iterable
import csv as _csv

import h5py
import numpy as np

from .schema import WaveformSample, T_SAMPLES, N_CHANNELS
from .adapters.base import BaseAdapter

log = logging.getLogger(__name__)


CSV_FIELDS = [
    "hdf5_index", "sample_id", "category", "sampling_category", "split",
    "source_file", "p_arrival_sample", "s_arrival_sample", "ps_diff_samples",
    "polarity", "content_hash",
]


def _content_hash(wf: np.ndarray, sample_id: str) -> str:
    """Stable 16-hex hash of (waveform bytes + sample_id). Used for dedup
    and parity validation against the legacy H5 files."""
    h = hashlib.blake2b(digest_size=8)
    h.update(wf.tobytes())
    h.update(sample_id.encode("utf-8"))
    return h.hexdigest()


def _format_picks(picks: Iterable[int]) -> str:
    """Match the legacy CSV format: '[]' or '[1234]' or '[589, 3286, 6090]'."""
    return "[" + ", ".join(str(int(p)) for p in picks) + "]"


def build_h5(
    adapter: BaseAdapter,
    out_h5: Path,
    out_csv: Path,
    *,
    chunk_size: int = 1024,
    max_samples: Optional[int] = None,
    progress: bool = True,
    overwrite: bool = False,
) -> dict:
    """Stream samples from `adapter` into HDF5 + CSV.

    Args:
        adapter      : iterator of WaveformSample.
        out_h5       : path to .h5 (will be created or overwritten if `overwrite`).
        out_csv      : path to metadata.csv.
        chunk_size   : H5 resize granularity (samples per resize).
        max_samples  : optional global cap (sums over splits).
        progress     : log a heartbeat every chunk_size samples.
        overwrite    : if False and either output exists, raise.

    Returns:
        dict with per-split sample counts.
    """
    out_h5 = Path(out_h5)
    out_csv = Path(out_csv)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if (out_h5.exists() or out_csv.exists()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing {out_h5} or {out_csv} "
            f"(pass overwrite=True)"
        )

    counts: dict[str, int] = {}     # split -> count
    write_idx: dict[str, int] = {}  # split -> next H5 row index

    h5 = h5py.File(out_h5, "w")
    csv_f = open(out_csv, "w", newline="")
    csv_w = _csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    csv_w.writeheader()

    # Per-split waveform datasets are created lazily on first sample.
    ds_per_split: dict[str, h5py.Dataset] = {}

    n_total = 0
    try:
        for sample in adapter:
            assert isinstance(sample, WaveformSample), type(sample)

            cat = sample.category
            split = sample.split

            # Lazy dataset creation
            split_key = f"{cat}/{split}"
            if split_key not in ds_per_split:
                grp = h5.require_group(f"{cat}/{split}")
                ds = grp.create_dataset(
                    "waveforms",
                    shape=(0, N_CHANNELS, T_SAMPLES),
                    maxshape=(None, N_CHANNELS, T_SAMPLES),
                    chunks=(1, N_CHANNELS, T_SAMPLES),
                    dtype=np.float32,
                    compression=None,
                )
                ds_per_split[split_key] = ds
                write_idx[split_key] = 0
                counts[split_key] = 0

            ds = ds_per_split[split_key]
            row_idx = write_idx[split_key]

            # Resize ahead of write
            new_size = row_idx + 1
            if new_size > ds.shape[0]:
                ds.resize((new_size, N_CHANNELS, T_SAMPLES))

            ds[row_idx] = sample.waveform

            # CSV row
            content_hash = sample.content_hash or _content_hash(sample.waveform, sample.sample_id)
            ps_diff = sample.ps_diff_samples
            csv_w.writerow({
                "hdf5_index":         row_idx,
                "sample_id":          sample.sample_id,
                "category":           cat,
                "sampling_category":  sample.sampling_category or cat,
                "split":              split,
                "source_file":        sample.source_file,
                "p_arrival_sample":   _format_picks(sample.p_arrival_samples),
                "s_arrival_sample":   _format_picks(sample.s_arrival_samples),
                "ps_diff_samples":    "" if ps_diff is None else int(ps_diff),
                "polarity":           sample.polarity,
                "content_hash":       content_hash,
            })

            write_idx[split_key] = row_idx + 1
            counts[split_key] = write_idx[split_key]
            n_total += 1

            if progress and n_total % chunk_size == 0:
                log.info(
                    "build_h5[%s/%s]: wrote %d samples; per-split %s",
                    adapter.name if hasattr(adapter, "name") else "?",
                    adapter.category if hasattr(adapter, "category") else "?",
                    n_total, counts,
                )

            if max_samples is not None and n_total >= max_samples:
                break

    finally:
        h5.close()
        csv_f.close()

    log.info("build_h5 done: %d total | per-split counts: %s", n_total, counts)
    return counts
