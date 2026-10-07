"""RockNet noise adapter reading from Luhu_dataset.h5.

Source layout (per RockNet ``DIR_README.md`` and the legacy 90 s H5):

    <rocknet_root>/                    # RockNet data/ directory
        Luhu_hdf5/Luhu_dataset.h5      # SAC-style HDF5 (see hierarchy below)
        metadata/partition/            # per-class train/val/test event_id partitions
            rockfall_partition.npy
            car_partition.npy
            engineering_partition.npy
            eq_partition.npy

The dataset is published by Liao et al. (2023) and contains continuous data
from the Luhu rockfall test site. Top-level groups in the HDF5:

    EQ                          # earthquake-labelled — *excluded* from noise
    RF_multiple_sta_label       # rockfall (used as noise)
    RF_single_sta_label         # rockfall (used as noise)
    car_multiple_sta_label      # car-induced signal (used as noise)
    engineering                 # engineering-machine signal (excluded — not in legacy v1)

Hierarchy under each group:
    {category}/{event_id}/{TW.STA.CHAN.LOC.event_id}/data    : (npts,) float32
                                                  /delta   : ()          float32
                                                  /...

For each (category, event_id, station) triple we pull the matching E/N/Z waveforms
and emit one 9000-sample window. RockNet "noise" therefore covers rockfall and
car-induced signals — these are non-earthquake but *not* silent — and they are
labelled as ``noise`` for training purposes (true negatives for the EQ branch).

Pad/window strategy:
    - Source trace lengths range ~12 000-90 000 samples at 100 Hz.
    - If trace ≥ 9 000: random-slice a 9 000-sample window.
    - If trace <  9 000: front-pad-only with spectrum-matched noise to reach
      9 000 samples (legacy P05 noise convention — keeps real signal aligned
      to the right edge of the window).

Splits come from the published ``*_partition.npy`` files. RockNet ships its
own train/val/test partitions and we honor them by default. Events absent
from the published partition fall back to a deterministic 3-way Bernoulli
draw so no row is silently dropped.

Audit fixes applied (2026-04-27):

* Use the published RockNet ``partition.npy`` files for split assignment
  whenever possible (``use_existing_partition=True``). Fall back to
  :func:`bernoulli_3way` only for events absent from any partition slot.
* The default ``skip_test_split=False`` means test rows ARE emitted;
  the smoke test confirms they appear in per-split counts so they are
  counted, not silently filtered.
* RockNet noise has no P/S picks, so the ``int()`` -> ``int(round())``
  fix does not apply here — there is nothing to parse.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np

from .base import BaseAdapter
from ..schema import (
    WaveformSample, T_SAMPLES, N_CHANNELS, SAMPLING_RATE_HZ,
)
from ..splits import bernoulli_3way
from redpan_motion.utils.waveform import find_reference_signal, generate_matching_noise


log = logging.getLogger(__name__)


# Group names in Luhu_dataset.h5 used as RED-PAN ``noise`` content.
# 2026-05-11: added "engineering" — anthropogenic engineering-machine signals
# from the same Luhu monitoring site (280 events). Useful as another "true
# negative" class for the EQ branch alongside rockfall (RF_*) + cars.
_NOISE_CATEGORIES = (
    "RF_multiple_sta_label",
    "RF_single_sta_label",
    "car_multiple_sta_label",
    "engineering",
)

# Filename prefix in the partition dir matching each H5 category.
_PARTITION_PREFIX = {
    "RF_multiple_sta_label": "rockfall",
    "RF_single_sta_label":   "rockfall",
    "car_multiple_sta_label": "car",
    "engineering":            "engineering",
}


def _load_partition(partition_root: Path, prefix: str) -> dict[str, set[str]]:
    """Load ``{prefix}_partition.npy`` and return ``{split: set(event_ids)}``.

    Missing or unreadable partitions return an empty mapping — those events
    fall back to :func:`bernoulli_3way` instead of being dropped.
    """
    p = partition_root / f"{prefix}_partition.npy"
    if not p.is_file():
        log.warning("RockNet: missing partition file %s", p)
        return {}
    try:
        raw = np.load(p, allow_pickle=True).item()
    except Exception as exc:   # noqa: BLE001 — we want any load failure to skip
        log.warning("RockNet: failed to load %s: %s", p, exc)
        return {}
    if not isinstance(raw, dict):
        log.warning("RockNet: partition %s is not a dict (got %s)", p, type(raw))
        return {}

    out: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for split_name in ("train", "val", "test"):
        ids = raw.get(split_name, [])
        for eid in ids:
            out[split_name].add(str(eid))
    return out


def _stations_for_event(event_grp: h5py.Group) -> dict[str, dict[str, str]]:
    """Group waveform IDs in an event by station, with E/N/Z component lookup.

    Source IDs look like ``TW.LH01.EHE.00.2019056113341``. We bucket by the
    second token (station code) and key components by the last char of the
    third token (E/N/Z).
    """
    by_station: dict[str, dict[str, str]] = {}
    for wf_id in event_grp.keys():
        parts = wf_id.split(".")
        if len(parts) < 4:
            continue
        sta = parts[1]
        comp = parts[2][-1] if parts[2] else ""
        if comp not in ("E", "N", "Z"):
            continue
        by_station.setdefault(sta, {})[comp] = wf_id
    return {k: v for k, v in by_station.items() if all(c in v for c in "ENZ")}


class RockNetNoiseAdapter(BaseAdapter):
    """RockNet noise — rockfall + car-induced signals from Luhu_dataset.h5.

    Args:
        luhu_h5                : path to ``Luhu_dataset.h5``.
        partition_root         : directory containing ``{rockfall,car}_partition.npy``.
        seed                   : RNG seed for slicing offsets, order shuffles,
                                 and Bernoulli fallback splits.
        skip_test_split        : if True, drop the ``test`` partition from output.
                                 Defaults False — schema accepts ``test``, and
                                 the published partition reserves ~6% for test
                                 (rockfall) up to ~28% (eq), so they should
                                 normally be emitted and counted.
        use_existing_partition : if True (default), look up each event in the
                                 published ``partition.npy`` files. If an event
                                 is missing, fall back to a deterministic
                                 :func:`bernoulli_3way` draw seeded by
                                 ``seed``. Set to False to ignore the
                                 published partition entirely.
        p_train / p_val        : Bernoulli fallback proportions (only used
                                 when ``use_existing_partition=False`` or for
                                 events missing from the published partition).

    The category produced is always ``noise`` per the project's category
    taxonomy (no P/S picks; ``polarity`` is empty).
    """

    name = "RockNet"
    category = "noise"

    def __init__(
        self,
        luhu_h5: Path,
        partition_root: Path,
        *,
        seed: int = 42,
        skip_test_split: bool = False,
        use_existing_partition: bool = True,
        p_train: float = 0.70,
        p_val: float = 0.15,
        windows_per_trace: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.luhu_h5 = Path(luhu_h5)
        self.partition_root = Path(partition_root)
        self.seed = seed
        self.skip_test_split = skip_test_split
        self.use_existing_partition = use_existing_partition
        self.p_train = p_train
        self.p_val = p_val
        # 2026-05-11: take N random non-coincident 9000-sample windows per
        # (event, station) trace instead of just one. Mirrors the legacy v1
        # pipeline's ~3 windows per trace and makes RockNet a more substantive
        # noise contributor (~3× more samples). All windows from the same
        # trace get distinct random start offsets — minor overlap is OK; the
        # model treats them as separate noise samples.
        self.windows_per_trace = max(1, int(windows_per_trace))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_components(event_grp: h5py.Group, comps: dict[str, str]) -> Optional[np.ndarray]:
        """Stack E/N/Z component data from one station.

        Returns ``(npts, 3)`` channel-last float32, or None if any component
        cannot be read or component lengths disagree.
        """
        try:
            e = np.asarray(event_grp[comps["E"]]["data"], dtype=np.float32)
            n = np.asarray(event_grp[comps["N"]]["data"], dtype=np.float32)
            z = np.asarray(event_grp[comps["Z"]]["data"], dtype=np.float32)
        except (KeyError, OSError):
            return None
        if e.ndim != 1 or n.ndim != 1 or z.ndim != 1:
            return None
        npts = min(e.size, n.size, z.size)
        if npts < 200:
            return None
        return np.stack([e[:npts], n[:npts], z[:npts]], axis=-1)

    @staticmethod
    def _front_pad_to_target(
        arr: np.ndarray,
        target_npts: int = T_SAMPLES,
    ) -> np.ndarray:
        """Front-pad-only with spectrum-matched noise to ``target_npts``.

        Per-channel reference is the first non-flat 500-sample window of the
        trace itself (mirrors the legacy STEAD/P05 noise convention).
        """
        npts, n_ch = arr.shape
        pad_npts = target_npts - npts
        if pad_npts <= 0:
            return arr.astype(np.float32, copy=False).copy()

        out = np.zeros((target_npts, n_ch), dtype=np.float32)
        for ch in range(n_ch):
            ref = find_reference_signal(
                arr[:, ch], window_size=500, max_search=5000, min_unique=300,
            )
            front = generate_matching_noise(ref, pad_npts).astype(np.float32)
            out[:pad_npts, ch] = front
            out[pad_npts:, ch] = arr[:, ch]
        return out

    def _resolve_split(
        self,
        evt_id: str,
        cat_part: dict[str, set[str]],
        rng: np.random.Generator,
    ) -> Optional[str]:
        """Pick a split for an event, preferring the published partition.

        Returns ``None`` only when the published partition is the sole
        source of truth (``use_existing_partition=True`` AND the partition
        file was actually loaded) AND the event is absent — in which case
        the caller logs and skips. Otherwise falls back to a deterministic
        Bernoulli 3-way draw.
        """
        if self.use_existing_partition and cat_part:
            for s_name in ("train", "val", "test"):
                if str(evt_id) in cat_part.get(s_name, set()):
                    return s_name
            # Fall back to bernoulli rather than silent drop — preserves
            # noise volume when the partition is incomplete.
            return bernoulli_3way(rng, self.p_train, self.p_val)
        # No partition available (or disabled): always use bernoulli.
        return bernoulli_3way(rng, self.p_train, self.p_val)

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[WaveformSample]:
        log.info("RockNetNoiseAdapter: reading %s", self.luhu_h5)

        # Build per-category partition (event_id -> split) once.
        partitions: dict[str, dict[str, set[str]]] = {}
        for cat in _NOISE_CATEGORIES:
            prefix = _PARTITION_PREFIX[cat]
            partitions[cat] = _load_partition(self.partition_root, prefix)

        rng = np.random.default_rng(self.seed)

        n_yielded = 0
        n_fallback = 0
        with h5py.File(self.luhu_h5, "r") as hf:
            for cat in _NOISE_CATEGORIES:
                if cat not in hf:
                    log.warning("  group %s missing in HDF5; skipping", cat)
                    continue

                cat_grp = hf[cat]
                # Deterministic event order from the source (we shuffle with rng
                # below to spread RF/car traces across the build).
                evt_ids = list(cat_grp.keys())
                rng.shuffle(evt_ids)

                cat_part = partitions[cat]
                partition_loaded = bool(cat_part)

                for evt_id in evt_ids:
                    if self.max_samples is not None and n_yielded >= self.max_samples:
                        break

                    # Track whether this event is actually in the published
                    # partition, for the fallback counter.
                    in_partition = (
                        partition_loaded
                        and self.use_existing_partition
                        and any(
                            str(evt_id) in cat_part.get(s_name, set())
                            for s_name in ("train", "val", "test")
                        )
                    )

                    split = self._resolve_split(str(evt_id), cat_part, rng)
                    if split is None:
                        continue
                    if not in_partition and self.use_existing_partition:
                        n_fallback += 1
                    if self.skip_test_split and split == "test":
                        continue
                    if self.split_filter and split != self.split_filter:
                        continue

                    event_grp = cat_grp[evt_id]
                    stations = _stations_for_event(event_grp)
                    for sta_code, comps in stations.items():
                        if self.max_samples is not None and n_yielded >= self.max_samples:
                            break

                        wf = self._stack_components(event_grp, comps)
                        if wf is None:
                            continue
                        npts = wf.shape[0]

                        # Pick distinct random start offsets for windows_per_trace
                        # windows. If the trace is too short to fit (npts<T_SAMPLES),
                        # fall back to a single padded window.
                        if npts < T_SAMPLES:
                            starts: list[int | None] = [None]   # None → padded fallback
                        else:
                            max_start = npts - T_SAMPLES
                            n_w = self.windows_per_trace
                            if max_start == 0 or n_w == 1:
                                starts = [int(rng.integers(0, max_start + 1))]
                            else:
                                # Sample without replacement when the trace can
                                # support n_w distinct starts; otherwise sample
                                # with replacement (degenerate small-trace case).
                                if max_start + 1 >= n_w:
                                    starts = sorted(int(s) for s in rng.choice(
                                        max_start + 1, size=n_w, replace=False))
                                else:
                                    starts = sorted(int(rng.integers(0, max_start + 1))
                                                    for _ in range(n_w))

                        for start in starts:
                            if self.max_samples is not None and n_yielded >= self.max_samples:
                                break
                            if start is None:
                                sliced = self._front_pad_to_target(wf, T_SAMPLES)
                                sample_id = f"noise_{cat}_{evt_id}_{sta_code}_pad"
                            else:
                                sliced = wf[start:start + T_SAMPLES]
                                sample_id = (
                                    f"noise_{cat}_{evt_id}_{sta_code}_{start}"
                                )

                            wf_cf = sliced.T.astype(np.float32, copy=False).copy()
                            if not np.isfinite(wf_cf).all():
                                log.debug(
                                    "skip %s/%s/%s start=%s: non-finite after pad/slice",
                                    cat, evt_id, sta_code, start,
                                )
                                continue

                            yield WaveformSample(
                                sample_id=sample_id,
                                waveform=wf_cf,
                                category="noise",
                                sampling_category="noise",
                                split=split,  # type: ignore[arg-type]
                                source_file=str(self.luhu_h5),
                                p_arrival_samples=[],
                                s_arrival_samples=[],
                                polarity="",
                            )
                            n_yielded += 1

        if n_fallback:
            log.info(
                "RockNetNoiseAdapter: %d events used Bernoulli fallback "
                "(absent from published partition).",
                n_fallback,
            )
