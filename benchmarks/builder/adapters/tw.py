"""TW (Taiwan catalogue) adapters reading from per-event SAC sources.

The TW dataset is the largest training source. Sources:
  - Event traces (180 s, 18000 npts): ``TW_2012_2019_180s/{year}/{jday}/{evid}/*.sac``
  - Per-year metadata (P/S residuals): ``metadata_TW_2012_2019_180s/sel_{year}.txt``
  - Hourly noise traces: ``TW_noise_data/metadata/TW_noise/{date.hour}/*.sac``
  - Pre-computed pure-noise masks: ``TW_noise_data/metadata/pred_TW_noise/{date.hour}/*.mask``
  - Polarity labels (optional): ``TW_eq_data/first_motion/available_data.csv``

Channel order on disk and in the legacy 90 s H5 is **(E, N, Z)** — verified via
SAC channel suffix ``EHE/EHN/EHZ`` (so after ``wf.sort()`` we get ``[E, N, Z]``
i.e. ``wf[0]=E``, ``wf[1]=N``, ``wf[2]=Z``).

Audit-driven fixes applied here (raw-SAC path):
  * Bug 1 — flatness checks use Z = ``wf[2]`` (not E = ``wf[0]``).
  * Bug 4 — explicit ``len(wf) != 3`` guard after every ``wf.slice() / .trim()``.
  * Bug 10 — explicit ``len(wf) < 3`` guard after ``read(...) + wf.sort()``.
  * Bug 12 — never duplicate a channel into a missing slot; rows lacking a full
    3-component trio are skipped.
  * Polarity column compatibility — CSV may use ``polarity`` or ``p_polarity``;
    we emit into ``WaveformSample.polarity`` ('U' / 'D' / '').
  * 3-way splits — noise uses ``bernoulli_3way``; event categories use
    ``hash_split`` keyed on the per-event source identifier so all rows from one
    earthquake land in the same split (no event leakage).
  * ``int(round(float(x)))`` for arrival samples (no truncation bias).
"""
from __future__ import annotations

import gzip
import logging
import os
import pickle
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Iterable, Iterator, Optional, Set, Tuple

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read

from .base import BaseAdapter
from ..schema import N_CHANNELS, SAMPLING_RATE_HZ, T_SAMPLES, WaveformSample
from ..splits import bernoulli_3way, hash_split
from redpan_motion.utils.waveform import find_reference_signal, generate_matching_noise


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (kept module-private)
# ---------------------------------------------------------------------------
def _ps_diff_bin(ps_diff_samples: int) -> str:
    """P-S separation in samples → category bin (matches train_*.json keys)."""
    sec = ps_diff_samples / SAMPLING_RATE_HZ
    if sec < 5:   return "singleEQ_00-05s"
    if sec < 10:  return "singleEQ_05-10s"
    if sec < 15:  return "singleEQ_10-15s"
    if sec < 20:  return "singleEQ_15-20s"
    return "singleEQ_20s_plus"


def _polarity_from_legacy(s: object) -> str:
    """Map TW polarity coding to {'U', 'D', ''}.

    Accepts '+'/'-' (first_motion CSV) and '[1]'/'[-1]'/'[null]'
    (legacy 90 s H5 metadata). Anything else maps to ''.
    """
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    txt = str(s).strip()
    if txt in ("+", "U", "u", "1", "[1]", "+1", "[+1]"):
        return "U"
    if txt in ("-", "D", "d", "-1", "[-1]"):
        return "D"
    return ""


def _is_finite_z(arr: np.ndarray, min_unique_frac: float = 0.05) -> bool:
    """Audit Bug-1 guard: reject samples whose Z channel (``arr[2]``) is flat
    (constant or near-constant) or non-finite. Operates on channel-first array.
    """
    if arr.shape[0] < 3:
        return False
    z = arr[2]
    if not np.isfinite(z).all():
        return False
    n_unique = len(np.unique(np.round(z, decimals=6)))
    return n_unique >= max(2, int(min_unique_frac * len(z)))


def _stream_to_3xN(st) -> Optional[np.ndarray]:
    """Stack (E, N, Z) traces into shape (3, npts).

    Audit Bug-12 fix: refuse to fabricate a missing channel by duplication.
    """
    if len(st) < 3:
        return None
    npts = min(len(t.data) for t in st[:3])
    if npts < 1:
        return None
    return np.stack([st[i].data[:npts] for i in range(3)], axis=0).astype(np.float32)


def _load_tw_metadata(metadir: Path, ps_min_sec: float = 0.3,
                      ps_max_sec: float = 1e9) -> pd.DataFrame:
    """Concatenate per-year ``sel_{year}.txt`` files into one DataFrame."""
    frames = []
    for txt in sorted(metadir.glob("sel_*.txt")):
        df = pd.read_csv(
            txt, header=0,
            names=["year", "jday", "evid", "station", "channel",
                   "psres", "matched", "EQ_detected"],
        )
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[(df["psres"] > ps_min_sec) & (df["psres"] < ps_max_sec)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level probe function for ProcessPool warmup. Must be importable (not
# a closure) so it can be pickled across process boundaries. obspy's SAC
# parser is GIL-bound, so threading delivered no speedup in benchmarks
# (0.9×); processes are required for real parallelism here.
# ---------------------------------------------------------------------------
def probe_tw_row_alive(args: Tuple[str, dict]) -> bool:
    """ProcessPool worker entry point. ``args = (wf_dir_str, row_dict)``.

    Returns True iff ``_read_event_arr`` would succeed for this row. We
    only return a bool — the actual waveform is discarded — so the IPC
    payload is one byte per probe.
    """
    wf_dir_str, row = args
    return _read_event_arr(Path(wf_dir_str), row) is not None


# ---------------------------------------------------------------------------
# Pre-scan: which (year, jday, evid, station, channel) groups have all 3
# component SAC files on disk? Eliminates the per-row ``len(paths) != 3``
# rejection at SAC-read time, which was the dominant dead-rate contributor
# (~73% of rows) for the mosaic adapter.
# ---------------------------------------------------------------------------
PrescanKey = Tuple[str, str, str, str, str]   # (year, jday, evid, station, ch_root)


def _scan_one_event_dir(args: Tuple[str, str, str, Path]) -> Set[PrescanKey]:
    """Scan one ``{wf_dir}/{year}/{jday}/{evid}`` directory.

    Filenames have shape ``{net}.{station}.{chan_full}.{loc}.sac`` where
    ``chan_full`` is 3 chars (e.g. ``HHE`` / ``HHN`` / ``HHZ``). We bucket
    by (station, chan_full[:2]) and emit a key for every bucket containing
    exactly 3 files (one per component).
    """
    year, jday, evid, evdir = args
    counts: dict = defaultdict(int)
    try:
        with os.scandir(evdir) as it:
            for entry in it:
                name = entry.name
                if not name.endswith(".sac"):
                    continue
                parts = name.split(".")
                if len(parts) != 5:
                    continue
                station = parts[1]
                chan_full = parts[2]
                if len(chan_full) != 3:
                    continue
                counts[(station, chan_full[:2])] += 1
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return set()
    return {
        (year, jday, evid, st, ch)
        for (st, ch), n in counts.items()
        if n == 3
    }


def _prescan_tw_valid_keys(
    wf_dir: Path,
    dir_keys: Iterable[Tuple[str, str, str]],
    *,
    max_workers: int = 16,
    cache_path: Optional[Path] = None,
    use_cache: bool = True,
) -> Set[PrescanKey]:
    """Validate which metadata rows have all 3 SAC components on disk.

    For each unique ``(year, jday, evid)`` directory referenced by the
    metadata, scandir once and emit valid keys. Threaded because the work
    is dominated by syscall latency on per-event directories — no shared
    state needed in the worker, just file-name parsing.

    Caches the resulting set to ``cache_path`` (gz-pickled) so subsequent
    runs skip the walk. Cache key = ``str(wf_dir)``; invalidation is
    manual (delete the cache file).
    """
    wf_dir = Path(wf_dir)
    if cache_path is None:
        cache_path = wf_dir / ".tw_prescan_cache.pkl.gz"
    cache_path = Path(cache_path)

    if use_cache and cache_path.exists():
        try:
            with gzip.open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and cached.get("wf_dir") == str(wf_dir):
                keys = cached["keys"]
                log.info("TW pre-scan cache hit: %s (%d valid keys)",
                         cache_path, len(keys))
                return keys
            log.info("TW pre-scan cache mismatch (different wf_dir); rebuilding")
        except Exception as exc:
            log.warning("TW pre-scan cache read failed (%s); rebuilding", exc)

    unique_dirs = sorted({(y, j, e) for (y, j, e) in dir_keys})
    log.info("TW pre-scan: walking %d unique event dirs (%d threads)",
             len(unique_dirs), max_workers)
    t0 = time.time()
    valid: Set[PrescanKey] = set()
    args_list = [
        (y, j, e, wf_dir / y / f"{int(j):03d}" / e)
        for (y, j, e) in unique_dirs
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_scan_one_event_dir, a) for a in args_list]
        for i, fut in enumerate(as_completed(futures), 1):
            valid.update(fut.result())
            if i % 50000 == 0:
                log.info("  pre-scan progress: %d/%d dirs (%.0fs, %d valid keys)",
                         i, len(unique_dirs), time.time() - t0, len(valid))
    log.info("TW pre-scan: %d valid keys from %d dirs in %.1fs",
             len(valid), len(unique_dirs), time.time() - t0)

    if use_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(cache_path, "wb") as f:
                pickle.dump({"wf_dir": str(wf_dir), "keys": valid}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            log.info("TW pre-scan: cached to %s", cache_path)
        except Exception as exc:
            log.warning("TW pre-scan cache write failed: %s", exc)

    return valid


def _split_key_for_event(row: pd.Series) -> str:
    """Per-event split key: prefer ``evid`` (TW catalogue convention)."""
    for col in ("evid", "event_id", "source_id"):
        v = row.get(col)
        if v is not None and str(v):
            return str(v)
    return f"{row.get('year', '')}_{row.get('jday', '')}_{row.get('station', '')}"


def _load_polarity_lookup(catalog_csv: Optional[Path]) -> dict:
    """Read TW first_motion catalog into a (year_jday_evid_station_channel) → polarity map.

    Accepts either ``polarity`` (TW first_motion convention) or ``p_polarity``
    (CEED convention) as the column name.
    """
    if catalog_csv is None or not Path(catalog_csv).exists():
        return {}
    cat = pd.read_csv(catalog_csv)
    pol_col = "p_polarity" if "p_polarity" in cat.columns else "polarity"
    if pol_col not in cat.columns:
        return {}
    needed = {"year", "jday", "evid", "station", "channel", pol_col}
    if not needed.issubset(cat.columns):
        return {}
    cat = cat[cat[pol_col].astype(str).isin(["+", "-", "U", "D", "u", "d"])].copy()
    cat["__key"] = (
        cat["year"].astype(str) + "_" +
        cat["jday"].astype(str).str.zfill(3) + "_" +
        cat["evid"].astype(str) + "_" +
        cat["station"].astype(str) + "_" +
        cat["channel"].astype(str)
    )
    return dict(zip(cat["__key"], cat[pol_col].map(_polarity_from_legacy)))


def _read_event_arr(
    wfdir: Path, row: pd.Series,
) -> Optional[Tuple[np.ndarray, int, int]]:
    """Read one event SAC trio → (arr (3, N), p_npts, s_npts)."""
    year = str(row["year"])
    jday = int(row["jday"])
    evid = str(row["evid"])
    station = str(row["station"])
    channel = str(row["channel"])
    pattern = str(wfdir / year / f"{jday:03}" / evid /
                  f"*.{station}.{channel}?.??.sac")
    paths = sorted(glob(pattern))
    if len(paths) != 3:                         # Bug 10 (input side)
        return None
    try:
        st = read(pattern)
    except Exception:
        return None
    st.sort()
    if len(st) < 3:                             # Bug 10 (post-sort)
        return None
    if any(np.round(t.stats.sampling_rate).astype(int) != SAMPLING_RATE_HZ
           for t in st):
        return None
    hdr = st[0].stats.sac
    needed = ("t1", "t2", "t3", "t4")
    if not all(k in hdr for k in needed):
        return None
    if abs(hdr["t3"] - hdr["t1"]) >= 0.5 or abs(hdr["t4"] - hdr["t2"]) >= 1.5:
        return None
    starttime = st[0].stats.starttime
    tp_utc = starttime - hdr.b + hdr.t3
    ts_utc = starttime - hdr.b + hdr.t4
    p_npts = int(round(float((tp_utc - starttime) * SAMPLING_RATE_HZ)))
    s_npts = int(round(float((ts_utc - starttime) * SAMPLING_RATE_HZ)))
    arr = _stream_to_3xN(st)                    # Bug 12: no fabrication
    if arr is None:
        return None
    if not (0 < p_npts < s_npts < arr.shape[1]):
        return None
    if not _is_finite_z(arr):                   # Bug 1: Z flatness
        return None
    return arr, p_npts, s_npts


def _spectrum_pad_3c(arr: np.ndarray, p_npts: int,
                     front_pad_n: int, back_pad_n: int) -> Optional[np.ndarray]:
    """Pad both sides spectrum-matched per-channel; returns (3, npts+pads)."""
    n_ch, npts = arr.shape
    out = np.zeros((n_ch, front_pad_n + npts + back_pad_n), dtype=np.float32)
    for ch in range(n_ch):
        pre = arr[ch, :p_npts] if p_npts > 0 else arr[ch, :500]
        if len(pre) < 200:
            pre = arr[ch, :min(2000, npts)]
        front_ref = (
            find_reference_signal(pre, window_size=500,
                                  max_search=max(500, len(pre)),
                                  min_unique=100)
            if len(pre) >= 500 else pre
        )
        tail = arr[ch, max(0, npts - 500):]
        back_ref = tail if len(tail) >= 200 else front_ref
        try:
            front_noise = generate_matching_noise(front_ref, front_pad_n).astype(np.float32)
            back_noise = generate_matching_noise(back_ref, back_pad_n).astype(np.float32)
        except Exception:
            return None
        out[ch] = np.concatenate([front_noise, arr[ch], back_noise])
    return out if np.isfinite(out).all() else None


# ---------------------------------------------------------------------------
# Noise adapter
# ---------------------------------------------------------------------------
class TWNoiseAdapter(BaseAdapter):
    """TW noise: 1-hour SAC streams sliced to 9000-sample pure-noise windows.

    Pure-noise filtering uses pre-computed prediction ``.mask`` files: any
    candidate slice with mask probability > ``pred_threshold`` (or P/S above
    threshold) is rejected.
    """
    name = "TW"
    category = "noise"

    def __init__(
        self,
        wf_dir: Path,                  # metadata/TW_noise (per-hour station SACs)
        pred_dir: Path,                # metadata/pred_TW_noise (.mask / .P / .S)
        *,
        p_train: float = 0.70,
        p_val: float = 0.15,
        seed: int = 42,
        windows_per_hour: int = 3,
        pred_threshold: float = 0.1,
        max_search_iters: int = 20,
        split_salt: str = "TW-noise-v1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wf_dir = Path(wf_dir)
        self.pred_dir = Path(pred_dir)
        self.p_train = p_train
        self.p_val = p_val
        self.seed = seed
        self.windows_per_hour = windows_per_hour
        self.pred_threshold = pred_threshold
        self.max_search_iters = max_search_iters
        self.split_salt = split_salt

    def _iter_mask_files(self) -> Iterator[Path]:
        return iter(sorted(self.pred_dir.glob("*/*.mask")))

    def __iter__(self) -> Iterator[WaveformSample]:
        log.info("TWNoiseAdapter: scanning %s", self.pred_dir)
        all_masks = list(self._iter_mask_files())
        log.info("  found %d .mask prediction files", len(all_masks))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(all_masks)

        n_yielded = 0
        for mask_path in all_masks:
            if self.max_samples is not None and n_yielded >= self.max_samples:
                return
            try:
                for sample in self._iter_hour(mask_path, rng):
                    yield sample
                    n_yielded += 1
                    if self.max_samples is not None and n_yielded >= self.max_samples:
                        return
            except Exception as exc:
                log.debug("skip %s: %s", mask_path.name, exc)
                continue

    def _iter_hour(self, mask_path: Path, rng: np.random.Generator
                   ) -> Iterator[WaveformSample]:
        """Yield up to ``windows_per_hour`` pure-noise slices for one hour file."""
        # mask file: TW.A007.HL.10.2021.273.16.sac.mask
        stem = (mask_path.name[:-len(".sac.mask")] if mask_path.name.endswith(".sac.mask")
                else mask_path.stem)
        date_hour = mask_path.parent.name        # 2021.273.16
        try:
            yr, jday, hour = date_hour.split(".")
        except ValueError:
            return
        try:
            net, sta, chn_root, loc, *_ = stem.split(".")
        except ValueError:
            return

        wf_pattern = (self.wf_dir / date_hour /
                      f"{net}.{sta}.{chn_root}?.{loc}.{yr}.{jday}.{hour}.sac")
        if len(sorted(glob(str(wf_pattern)))) != 3:    # Bug 10 (input)
            return

        try:
            wf = read(str(wf_pattern))
        except Exception:
            return
        wf.sort()
        if len(wf) < 3:                                # Bug 10 (post-sort)
            return
        if any(np.round(t.stats.sampling_rate).astype(int) != SAMPLING_RATE_HZ
               for t in wf):
            return

        try:
            mask = read(str(mask_path)).sort()
            p = read(str(mask_path).replace(".mask", ".P")).sort()
            s = read(str(mask_path).replace(".mask", ".S")).sort()
        except Exception:
            return

        starttime = UTCDateTime(year=int(yr), julday=int(jday), hour=int(hour))
        for stream in (wf, mask, p, s):
            stream.trim(starttime=starttime, endtime=starttime + 3600,
                        nearest_sample=False)
            if len(stream) == 0:                       # all data clipped out
                return
        # Bug 4 (post-trim): check only waveform — mask / P / S are 1-trace
        # streams by construction.
        if len(wf) < 3:
            return

        pred_p = p[0].data
        pred_s = s[0].data
        pred_m = mask[0].data
        n_avail = min(len(pred_p), len(pred_s), len(pred_m))
        if n_avail < T_SAMPLES + 100:
            return
        st_pt_max = n_avail - T_SAMPLES

        # Hour-level deterministic 3-way split (independent of waveform draw rng)
        # — every window from this hour goes into the same split.
        hour_key = f"{net}.{sta}.{chn_root}.{loc}_{date_hour}"
        split = hash_split(hour_key, self.p_train, self.p_val, salt=self.split_salt)
        if self.split_filter and split != self.split_filter:
            return

        wf_E, wf_N, wf_Z = wf[0].data, wf[1].data, wf[2].data
        wf_n = min(len(wf_E), len(wf_N), len(wf_Z))
        if wf_n < T_SAMPLES:
            return

        for slot in range(self.windows_per_hour):
            start = -1
            for _ in range(self.max_search_iters):
                cand = int(rng.integers(0, st_pt_max + 1))
                if (pred_p[cand:cand + T_SAMPLES].max() <= self.pred_threshold and
                        pred_s[cand:cand + T_SAMPLES].max() <= self.pred_threshold and
                        pred_m[cand:cand + T_SAMPLES].max() <= self.pred_threshold):
                    start = cand
                    break
            if start < 0 or start + T_SAMPLES > wf_n:
                continue

            arr = np.stack([
                wf_E[start:start + T_SAMPLES],
                wf_N[start:start + T_SAMPLES],
                wf_Z[start:start + T_SAMPLES],
            ], axis=0).astype(np.float32)
            if not np.isfinite(arr).all():
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                if not np.isfinite(arr).all():
                    continue
            if not _is_finite_z(arr):                   # Bug 1
                continue

            sid = f"{net}.{sta}.{chn_root}.{loc}_{yr}.{jday}.{hour}_{slot+1:02d}"
            yield WaveformSample(
                sample_id=f"noise_{sid}",
                waveform=arr,
                category="noise",
                sampling_category="noise",
                split=split,
                source_file=str(mask_path.parent.parent / date_hour / f"{stem}.sac"),
                p_arrival_samples=[],
                s_arrival_samples=[],
                polarity="",
            )


# ---------------------------------------------------------------------------
# Event-bearing adapters share an iterator; per-class hooks customise slicing.
# ---------------------------------------------------------------------------
class _TWEventAdapterBase(BaseAdapter):
    """Common scaffolding for singleEQ / Ponly / Sonly / singleEQ_zeropad."""
    name = "TW"
    split_salt: str = "TW-event-v1"
    rng_offset: int = 0
    keep_polarity: bool = True

    def __init__(
        self,
        wf_dir: Path,
        meta_dir: Path,
        *,
        polarity_csv: Optional[Path] = None,
        p_train: float = 0.70,
        p_val: float = 0.15,
        seed: int = 42,
        ps_min_sec: float = 0.3,
        ps_max_sec: float = 60.0,
        stratify_ps_bins: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wf_dir = Path(wf_dir)
        self.meta_dir = Path(meta_dir)
        self.polarity_csv = Path(polarity_csv) if polarity_csv else None
        self.p_train = p_train
        self.p_val = p_val
        self.seed = seed
        self.ps_min_sec = ps_min_sec
        self.ps_max_sec = ps_max_sec
        self.stratify_ps_bins = stratify_ps_bins

    def _build_sample(
        self, row: pd.Series, arr: np.ndarray, p_npts: int, s_npts: int,
        rng: np.random.Generator, polarity: str, split: str, n_yielded: int,
    ) -> Optional[WaveformSample]:
        """Subclass hook — must return None or a valid WaveformSample."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[WaveformSample]:
        log.info("%s: loading metadata from %s",
                 self.__class__.__name__, self.meta_dir)
        df = _load_tw_metadata(self.meta_dir,
                               ps_min_sec=self.ps_min_sec,
                               ps_max_sec=self.ps_max_sec)
        log.info("  rows after psres filter: %d", len(df))
        df = df.sample(frac=1.0, random_state=self.seed + self.rng_offset).reset_index(drop=True)
        polarity_lookup = (_load_polarity_lookup(self.polarity_csv)
                           if self.keep_polarity else {})
        rng = np.random.default_rng(self.seed + self.rng_offset)

        # Stratification setup
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
            log.info("%s: stratify_ps_bins=True, per-bin cap=%d",
                     self.__class__.__name__, cap)

        n_yielded = 0
        n_skip_bin_full = 0
        for _, row in df.iterrows():
            if self.max_samples is not None and n_yielded >= self.max_samples:
                break

            split = hash_split(_split_key_for_event(row),
                               self.p_train, self.p_val, salt=self.split_salt)
            if self.split_filter and split != self.split_filter:
                continue

            data = _read_event_arr(self.wf_dir, row)
            if data is None:
                continue
            arr, p_npts, s_npts = data

            polarity = ""
            if self.keep_polarity:
                key = (f"{row['year']}_{int(row['jday']):03d}_{row['evid']}_"
                       f"{row['station']}_{row['channel']}")
                polarity = polarity_lookup.get(key, "")

            sample = self._build_sample(row, arr, p_npts, s_npts,
                                        rng, polarity, split, n_yielded)
            if sample is None:
                continue

            # Stratification: enforce per-bin cap before yield.
            if bin_caps:
                samp_cat = sample.sampling_category or ""
                if bin_counts.get(samp_cat, 0) >= bin_caps.get(samp_cat, 0):
                    n_skip_bin_full += 1
                    continue
                bin_counts[samp_cat] = bin_counts.get(samp_cat, 0) + 1

            try:
                yield sample
            except ValueError as exc:
                log.debug("skip %s: %s", sample.sample_id, exc)
                continue
            n_yielded += 1

        if bin_caps:
            log.info("%s per-bin yield: %s  (skip_bin_full=%d)",
                     self.__class__.__name__, dict(bin_counts), n_skip_bin_full)


# ---------------------------------------------------------------------------
# SingleEQ — full P + S window
# ---------------------------------------------------------------------------
class TWSingleEQAdapter(_TWEventAdapterBase):
    category = "singleEQ"
    split_salt = "TW-singleEQ-v1"
    rng_offset = 0
    keep_polarity = True

    def __init__(self, *args, min_p_position: int = 500,
                 max_p_position: int = 7500, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_p_position = min_p_position
        self.max_p_position = max_p_position

    def _build_sample(self, row, arr, p_npts, s_npts, rng,
                      polarity, split, n_yielded):
        ps_residual = s_npts - p_npts
        if ps_residual <= 0 or ps_residual >= T_SAMPLES:
            return None

        front_pad_n = self.max_p_position + 1000
        back_pad_n = T_SAMPLES + 1000
        padded = _spectrum_pad_3c(arr, p_npts, front_pad_n, back_pad_n)
        if padded is None:
            return None

        target_p = int(rng.integers(self.min_p_position, self.max_p_position + 1))
        slice_start = front_pad_n + p_npts - target_p
        slice_end = slice_start + T_SAMPLES
        if slice_start < 0 or slice_end > padded.shape[1]:
            return None
        wf_cf = padded[:, slice_start:slice_end].copy()
        new_p, new_s = target_p, target_p + ps_residual
        if not (0 <= new_p < T_SAMPLES and 0 <= new_s < T_SAMPLES):
            return None
        if not np.isfinite(wf_cf).all():
            return None

        sid = (f"{split}_{row['year']}_{int(row['jday']):03d}_"
               f"{row['evid']}_{row['station']}_{row['channel']}_{n_yielded:07d}")
        return WaveformSample(
            sample_id=sid,
            waveform=wf_cf.astype(np.float32, copy=False),
            category="singleEQ",
            sampling_category=_ps_diff_bin(ps_residual),
            split=split,
            source_file=str(self.wf_dir / str(row["year"]) /
                            f"{int(row['jday']):03d}" / str(row["evid"])),
            p_arrival_samples=[int(new_p)],
            s_arrival_samples=[int(new_s)],
            polarity=polarity,
        )


# ---------------------------------------------------------------------------
# Ponly — only P inside the 9000-sample window (S is past slice_end)
# ---------------------------------------------------------------------------
class TWPonlyAdapter(_TWEventAdapterBase):
    category = "Ponly"
    split_salt = "TW-Ponly-v1"
    rng_offset = 1
    keep_polarity = True

    def __init__(self, *args, min_p_position: int = 4000,
                 max_p_position: int = 8800, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_p_position = min_p_position
        self.max_p_position = max_p_position

    def _build_sample(self, row, arr, p_npts, s_npts, rng,
                      polarity, split, n_yielded):
        ps_residual = s_npts - p_npts
        front_pad_n = self.max_p_position + 1000
        back_pad_n = T_SAMPLES + 1000
        padded = _spectrum_pad_3c(arr, p_npts, front_pad_n, back_pad_n)
        if padded is None:
            return None

        ts_buffer = 20  # 0.2 s safety so S falls just past slice_end
        min_target_p = max(self.min_p_position,
                           T_SAMPLES + ts_buffer - ps_residual)
        if min_target_p > self.max_p_position:
            return None
        target_p = int(rng.integers(min_target_p, self.max_p_position + 1))
        slice_start = front_pad_n + p_npts - target_p
        slice_end = slice_start + T_SAMPLES
        if slice_start < 0 or slice_end > padded.shape[1]:
            return None

        wf_cf = padded[:, slice_start:slice_end].copy()
        new_p = target_p
        new_s = target_p + ps_residual
        if not (0 <= new_p < T_SAMPLES) or (0 <= new_s < T_SAMPLES):
            return None
        if not np.isfinite(wf_cf).all():
            return None

        sid = (f"{split}_{row['year']}_{int(row['jday']):03d}_"
               f"{row['evid']}_{row['station']}_{row['channel']}_Ponly_{n_yielded:07d}")
        return WaveformSample(
            sample_id=sid,
            waveform=wf_cf.astype(np.float32, copy=False),
            category="Ponly",
            sampling_category="Ponly",
            split=split,
            source_file=str(self.wf_dir / str(row["year"]) /
                            f"{int(row['jday']):03d}" / str(row["evid"])),
            p_arrival_samples=[int(new_p)],
            s_arrival_samples=[],
            polarity=polarity,
        )


# ---------------------------------------------------------------------------
# Sonly — only S inside the window (P is before slice_start)
# ---------------------------------------------------------------------------
class TWSonlyAdapter(_TWEventAdapterBase):
    category = "Sonly"
    split_salt = "TW-Sonly-v1"
    rng_offset = 2
    keep_polarity = False

    def __init__(self, *args, min_s_position: int = 200,
                 max_s_position: int = 8000, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_s_position = min_s_position
        self.max_s_position = max_s_position

    def _build_sample(self, row, arr, p_npts, s_npts, rng,
                      polarity, split, n_yielded):
        front_pad_n = T_SAMPLES + 1000
        back_pad_n = T_SAMPLES + 1000
        padded = _spectrum_pad_3c(arr, p_npts, front_pad_n, back_pad_n)
        if padded is None:
            return None

        p_in_padded = front_pad_n + p_npts
        s_in_padded = front_pad_n + s_npts
        target_s = int(rng.integers(self.min_s_position, self.max_s_position + 1))
        slice_start = s_in_padded - target_s
        slice_end = slice_start + T_SAMPLES
        if slice_start <= p_in_padded:               # P must be excluded
            return None
        if slice_start < 0 or slice_end > padded.shape[1]:
            return None

        wf_cf = padded[:, slice_start:slice_end].copy()
        if not (0 <= target_s < T_SAMPLES) or not np.isfinite(wf_cf).all():
            return None

        sid = (f"{split}_{row['year']}_{int(row['jday']):03d}_"
               f"{row['evid']}_{row['station']}_{row['channel']}_Sonly_{n_yielded:07d}")
        return WaveformSample(
            sample_id=sid,
            waveform=wf_cf.astype(np.float32, copy=False),
            category="Sonly",
            sampling_category="Sonly",
            split=split,
            source_file=str(self.wf_dir / str(row["year"]) /
                            f"{int(row['jday']):03d}" / str(row["evid"])),
            p_arrival_samples=[],
            s_arrival_samples=[int(target_s)],
            polarity="",
        )


# ---------------------------------------------------------------------------
# SingleEQ + zeropad augmentation
# ---------------------------------------------------------------------------
def _apply_zeropad_aug(arr: np.ndarray, rng: np.random.Generator,
                       p_drop_channel: float = 0.3,
                       p_flat: float = 0.2) -> np.ndarray:
    """Channel-drop + zero-pad-segment augmentation matching the legacy
    P02_TFRecord_SingleEQ_*_zeropad behaviour. Operates on a fresh copy."""
    out = arr.copy()
    n_ch, npts = out.shape
    if rng.random() < p_drop_channel:
        choice = int(rng.integers(0, 3))         # E only / N only / both E+N
        if choice == 0:
            out[0] = 0
        elif choice == 1:
            out[1] = 0
        else:
            out[0] = 0
            out[1] = 0
    if rng.random() < p_flat:
        n_segments = int(rng.integers(1, 4))
        for _ in range(n_segments):
            seg_len = int(rng.integers(50, 201))
            max_start = npts - seg_len - 100
            if max_start <= 100:
                continue
            seg_start = int(rng.integers(100, max_start + 1))
            seg_chn = int(rng.integers(0, 4))
            if seg_chn < 3:
                out[seg_chn, seg_start:seg_start + seg_len] = 0
            else:
                out[:, seg_start:seg_start + seg_len] = 0
    return out


class TWSingleEQZeropadAdapter(_TWEventAdapterBase):
    category = "singleEQ_zeropad"
    split_salt = "TW-singleEQ_zeropad-v1"
    rng_offset = 3
    keep_polarity = False                # legacy convention: zero-pad nukes Z polarity

    def __init__(self, *args, min_p_position: int = 500,
                 max_p_position: int = 7500,
                 p_drop_channel: float = 0.3, p_flat: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_p_position = min_p_position
        self.max_p_position = max_p_position
        self.p_drop_channel = p_drop_channel
        self.p_flat = p_flat

    def _build_sample(self, row, arr, p_npts, s_npts, rng,
                      polarity, split, n_yielded):
        ps_residual = s_npts - p_npts
        if ps_residual <= 0 or ps_residual >= T_SAMPLES:
            return None

        front_pad_n = self.max_p_position + 1000
        back_pad_n = T_SAMPLES + 1000
        padded = _spectrum_pad_3c(arr, p_npts, front_pad_n, back_pad_n)
        if padded is None:
            return None

        target_p = int(rng.integers(self.min_p_position, self.max_p_position + 1))
        slice_start = front_pad_n + p_npts - target_p
        slice_end = slice_start + T_SAMPLES
        if slice_start < 0 or slice_end > padded.shape[1]:
            return None
        wf_cf = padded[:, slice_start:slice_end].copy()
        new_p, new_s = target_p, target_p + ps_residual
        if not (0 <= new_p < T_SAMPLES and 0 <= new_s < T_SAMPLES):
            return None

        wf_cf = _apply_zeropad_aug(wf_cf, rng,
                                   p_drop_channel=self.p_drop_channel,
                                   p_flat=self.p_flat)
        if not np.isfinite(wf_cf).all():
            return None

        sid = (f"{split}_{row['year']}_{int(row['jday']):03d}_"
               f"{row['evid']}_{row['station']}_{row['channel']}_ZP_{n_yielded:07d}")
        return WaveformSample(
            sample_id=sid,
            waveform=wf_cf.astype(np.float32, copy=False),
            category="singleEQ_zeropad",
            sampling_category=_ps_diff_bin(ps_residual),
            split=split,
            source_file=str(self.wf_dir / str(row["year"]) /
                            f"{int(row['jday']):03d}" / str(row["evid"])),
            p_arrival_samples=[int(new_p)],
            s_arrival_samples=[int(new_s)],
            polarity="",
        )
