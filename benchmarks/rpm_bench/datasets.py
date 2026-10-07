"""Record iterators for the static runs. Copied from the benchmark harness that produced the
manuscript numbers (``benchmark_unified_filt.py``); only the data locations and the input filter
are now arguments instead of hard-coded paths and environment variables, and missing inputs are
reported instead of being skipped silently. Which records are read is unchanged.

Each iterator yields ``(wf, info)``: the preprocessed record ``(T, 3)`` (E, N, Z) and
``info = {evid, label_type, labelP_sec, labelS_sec, polarity_label}``.
"""

from __future__ import annotations

import ast as _ast
import logging
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .constants import DT, SENTINEL
from .static import preprocess_array, restore_taper

log = logging.getLogger(__name__)


def _load_stead_test_ids(test_ids_path: Path) -> set[str]:
    """Load the pre-defined STEAD test trace_name set from test.npy."""
    # test.npy of the STEAD release is an object array of trace names, so it needs
    # allow_pickle; check its md5 (README, Data) before loading a copy from elsewhere.
    arr = np.load(test_ids_path, allow_pickle=True)
    return set(arr.tolist())


def iter_stead_eq(
    merge_csv: Path,
    merge_hdf5: Path,
    max_eq: int = 0,
    test_only: bool = True,
    filt: str = "bp145",
) -> Iterable[tuple[np.ndarray, dict]]:
    df = pd.read_csv(merge_csv, low_memory=False)
    eq = df[df["trace_category"] == "earthquake_local"]
    if test_only:
        ids = _load_stead_test_ids(Path(merge_csv).parent / "test.npy")
        eq = eq[eq["trace_name"].isin(ids)]
    n = 0
    with h5py.File(merge_hdf5, "r") as hf:
        for _, row in eq.iterrows():
            if max_eq and n >= max_eq:
                break
            tn = str(row["trace_name"])
            try:
                wf = np.array(hf["data"][tn]).astype(np.float32)
                p = int(round(float(row["p_arrival_sample"])))
                s = int(round(float(row["s_arrival_sample"])))
            except (KeyError, TypeError, ValueError):
                continue
            if wf.shape != (6000, 3) or not (0 <= p < s < 6000):
                continue
            yield (
                preprocess_array(wf, filt=filt),
                dict(
                    evid=tn,
                    label_type="earthquake",
                    labelP_sec=p * DT,
                    labelS_sec=s * DT,
                    polarity_label="",
                ),
            )
            n += 1


def iter_stead_noise(
    merge_csv: Path,
    merge_hdf5: Path,
    max_noise: int = 0,
    test_only: bool = True,
    filt: str = "bp145",
) -> Iterable[tuple[np.ndarray, dict]]:
    df = pd.read_csv(merge_csv, low_memory=False)
    nz = df[df["trace_category"] == "noise"]
    if test_only:
        ids = _load_stead_test_ids(Path(merge_csv).parent / "test.npy")
        nz = nz[nz["trace_name"].isin(ids)]
    n = 0
    with h5py.File(merge_hdf5, "r") as hf:
        for _, row in nz.iterrows():
            if max_noise and n >= max_noise:
                break
            tn = str(row["trace_name"])
            try:
                wf = np.array(hf["data"][tn]).astype(np.float32)
            except KeyError:
                continue
            if wf.shape != (6000, 3):
                continue
            yield (
                preprocess_array(wf, filt=filt),
                dict(
                    evid=tn,
                    label_type="noise",
                    labelP_sec=SENTINEL,
                    labelS_sec=SENTINEL,
                    polarity_label="",
                ),
            )
            n += 1


def iter_stead_noise_test(
    noise_dir: Path, crop_start: int = 0, max_noise: int = 0, filt: str = "bp145"
) -> Iterable[tuple[np.ndarray, dict]]:
    """STEAD test noise from the 90 s archive. ``crop_start=3000`` yields the
    original 60 s trace (for 60 s models and SeisBench baselines); 0 yields all 9,000.
    The crop is applied before preprocessing, as for a native 60 s STEAD trace."""
    c0 = int(crop_start)
    md = pd.read_csv(Path(noise_dir) / "STEAD_dataset_90s_noise_test_metadata.csv")
    with h5py.File(Path(noise_dir) / "STEAD_dataset_90s_noise_test.h5", "r") as hf:
        ds = hf["noise/test/waveforms"]
        for n, row in enumerate(md.itertuples(index=False)):
            if max_noise and n >= max_noise:
                break
            wf = ds[int(row.hdf5_index)][:, c0:].T.astype(np.float32)  # (T, 3) E, N, Z
            yield (
                preprocess_array(wf, filt=filt),
                dict(
                    evid=str(row.sample_id).removeprefix("noise_"),
                    label_type="noise",
                    labelP_sec=SENTINEL,
                    labelS_sec=SENTINEL,
                    polarity_label="",
                ),
            )


def iter_geonet_eq(
    data_root: Path,
    max_eq: int = 0,
    filt: str = "bp145",
) -> Iterable[tuple[np.ndarray, dict]]:
    """Yield GeoNet test earthquake traces. Waveforms are (T, 3) float32 with
    T variable (~27000 samples = 270 s). p_sample/s_sample are absolute sample
    indices into the loaded waveform."""
    meta_csv = data_root / "metadata.csv"
    df = pd.read_csv(meta_csv, low_memory=False)
    n = 0
    for _, row in df.iterrows():
        if max_eq and n >= max_eq:
            break
        tn = str(row["trace_name"])
        npy_path = data_root / "waveforms" / f"{tn}.npy"
        try:
            wf = np.load(npy_path).astype(np.float32)
            p = int(row["p_sample"])
            s = int(row["s_sample"])
        except (OSError, FileNotFoundError, KeyError, TypeError, ValueError):
            continue
        if wf.ndim != 2 or wf.shape[1] != 3 or not (0 <= p < s < wf.shape[0]):
            continue
        yield (
            preprocess_array(wf, filt=filt),
            dict(
                evid=tn,
                label_type="earthquake",
                labelP_sec=p * DT,
                labelS_sec=s * DT,
                polarity_label="",
            ),
        )
        n += 1


def iter_geonet_noise(
    data_root: Path,
    max_noise: int = 0,
    apply_taper_fix: bool = True,
    filt: str = "bp145",
) -> Iterable[tuple[np.ndarray, dict]]:
    """Yield GeoNet test noise traces (~12000 samples = 120 s).

    GeoNet noise traces ship with a Tukey-style amplitude envelope already
    baked in (sample 0 ≈ 1e-9; mid-trace ≈ 1e-4). Under sliding inference
    this onset-shaped rise mimics a P arrival and triggers the model.
    With ``apply_taper_fix=True`` (default), the tapered edges are detected
    and replaced with spectrum-matched noise from the un-tapered middle
    *after* ``preprocess_array`` (post-bandpass)."""
    meta_csv = data_root / "metadata_noise.csv"
    df = pd.read_csv(meta_csv, low_memory=False)
    n = 0
    for _, row in df.iterrows():
        if max_noise and n >= max_noise:
            break
        tn = str(row["trace_name"])
        npy_path = data_root / "noise_waveforms" / f"{tn}.npy"
        try:
            wf = np.load(npy_path).astype(np.float32)
        except (OSError, FileNotFoundError):
            continue
        if wf.ndim != 2 or wf.shape[1] != 3:
            continue
        wf_pre = preprocess_array(wf, filt=filt)
        if apply_taper_fix:
            wf_pre = restore_taper(wf_pre)
        yield (
            wf_pre,
            dict(
                evid=tn,
                label_type="noise",
                labelP_sec=SENTINEL,
                labelS_sec=SENTINEL,
                polarity_label="",
            ),
        )
        n += 1


def _parse_arrival(s) -> list:
    """Arrival samples of a metadata cell such as ``"[752]"``; ``[]`` if empty or unparsable."""
    if isinstance(s, str) and s.strip():
        try:
            v = _ast.literal_eval(s)
            return v if isinstance(v, list) else [int(v)]
        except (ValueError, SyntaxError):
            return []
    return []


def _pick_split(splits_present: set) -> str:
    """Return preferred split: 'test' if present else 'val' else 'train'."""
    for s in ("test", "val", "train"):
        if s in splits_present:
            return s
    return ""


def _h5_yield_one(h5, csv_row, dataset_label, dt=DT, filt="bp145"):
    """Load one waveform from H5 + CSV row, return (wf, info) or None."""
    try:
        idx = int(csv_row["hdf5_index"])
        cat = str(csv_row["category"])  # may have nested path like "singleEQ/Ponly_ps10_below"
        split = str(csv_row["split"])
        p_list = _parse_arrival(csv_row.get("p_arrival_sample", ""))
        s_list = _parse_arrival(csv_row.get("s_arrival_sample", ""))
    except (KeyError, ValueError, TypeError):
        return None
    h5_path = f"/{cat}/{split}/waveforms"
    if h5_path not in h5:
        return None
    wf3T = h5[h5_path][idx]  # (3, 9000)
    if wf3T.shape != (3, 9000):
        return None
    wf = wf3T.T.astype(np.float32, copy=False)  # (9000, 3)
    p_sample = p_list[0] if p_list else None
    s_sample = s_list[0] if s_list else None
    # the original also tested cat.endswith("noise"), which implies the first test here
    is_noise = ("noise" in cat.split("/")[-1]) or (p_sample is None and s_sample is None)
    if is_noise:
        info = dict(
            evid=str(csv_row.get("sample_id", f"{dataset_label}_{idx}")),
            label_type="noise",
            labelP_sec=SENTINEL,
            labelS_sec=SENTINEL,
            polarity_label="",
        )
    else:
        if (
            p_sample is None
            or s_sample is None
            or not (0 <= p_sample < 9000 and 0 <= s_sample < 9000)
        ):
            return None
        info = dict(
            evid=str(csv_row.get("sample_id", f"{dataset_label}_{idx}")),
            label_type="earthquake",
            labelP_sec=p_sample * dt,
            labelS_sec=s_sample * dt,
            polarity_label="",
        )
    return preprocess_array(wf, filt=filt), info


def iter_h5_dataset(
    root: Path, kind: str, max_n: int = 0, filt: str = "bp145", rows: str | None = None
) -> Iterable[tuple[np.ndarray, dict]]:
    """Yield (preprocessed_wf, info) from a dataset's category H5(s).

    `kind` is "eq" or "noise". For "eq" we pick singleEQ + Ponly + Sonly H5
    files; for "noise" we pick the noise H5 file. We use the test split if
    any contributing file has it, else val.

    A dataset without earthquake files or splits is an error; one without noise files (CEED,
    CREW, ROMPLUS) yields nothing and logs a warning. ``rows`` ("start:stop") slices the split
    of each file before ``max_n`` is applied.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    if kind == "eq":
        patterns = ("*singleEQ.h5", "*singleEQ_zeropad.h5", "*Ponly.h5", "*Sonly.h5")
    elif kind == "noise":
        patterns = ("*noise.h5",)
    else:
        raise ValueError(f"kind must be 'eq' or 'noise', got {kind!r}")

    files = sorted({p for pat in patterns for p in root.glob(pat)})
    if not files:
        msg = f"no {kind} HDF5 files ({', '.join(patterns)}) in {root}"
        if kind == "eq":
            raise FileNotFoundError(msg + "; check config key h5_root / RPM_BENCH_H5_ROOT")
        log.warning("%s; no noise records from this dataset", msg)
        return

    # Decide split: test if any file has it, else val.
    splits_present = set()
    for h5p in files:
        with h5py.File(h5p, "r") as h:
            for top in h.keys():
                grp = h[top]
                if not isinstance(grp, h5py.Group):
                    continue

                # Top-level may be category like 'singleEQ' or have nested like
                # 'singleEQ/Ponly_ps10_below'. Splits live one or two levels in.
                def collect(g, depth=0):
                    if depth > 3:
                        return
                    for k in g.keys():
                        if k in ("train", "val", "test"):
                            splits_present.add(k)
                        elif isinstance(g[k], h5py.Group):
                            collect(g[k], depth + 1)

                collect(grp)
    split = _pick_split(splits_present)
    if not split:
        msg = f"no train/val/test split groups in {[str(f) for f in files]}"
        if kind == "eq":
            raise ValueError(msg)
        log.warning("%s; no noise records from this dataset", msg)
        return

    n = 0
    for h5p in files:
        csv_path = h5p.with_suffix("").as_posix() + "_metadata.csv"
        if not Path(csv_path).exists():
            log.warning("metadata CSV %s not found; skipping %s", csv_path, h5p)
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df["split"] == split]
        if rows:  # "start:stop" — one chunk of a large split, for parallel runs
            r0, r1 = (int(v) if v else None for v in rows.split(":"))
            df = df.iloc[r0:r1]
        if len(df) == 0:
            continue
        with h5py.File(h5p, "r") as h:
            for _, row in df.iterrows():
                if max_n and n >= max_n:
                    return
                out = _h5_yield_one(h, row, f"{root.name}_{h5p.stem}", filt=filt)
                if out is not None:
                    yield out
                    n += 1


def iter_h5_noise_pool(
    h5_root: Path, pool: str, max_noise: int = 0, filt: str = "bp145"
) -> Iterable[tuple[np.ndarray, dict]]:
    """Test split of a pooled-noise HDF5 (GeoNet / RockNet), full 90 s record."""
    root = Path(h5_root) / pool
    md = pd.read_csv(root / f"{pool}_dataset_90s_noise_metadata.csv", low_memory=False)
    md = md[md["split"] == "test"]
    with h5py.File(root / f"{pool}_dataset_90s_noise.h5", "r") as hf:
        for n, row in enumerate(md.itertuples(index=False)):
            if max_noise and n >= max_noise:
                break
            wf = hf[f"/{row.category}/test/waveforms"][int(row.hdf5_index)].T.astype(np.float32)
            yield (
                preprocess_array(wf, filt=filt),
                dict(
                    evid=f"{pool}_{int(row.hdf5_index)}",
                    label_type="noise",
                    labelP_sec=SENTINEL,
                    labelS_sec=SENTINEL,
                    polarity_label="",
                ),
            )
