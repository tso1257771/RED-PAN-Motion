"""
Dataset for RED-PAN 90s model (multi-dataset H5 with polarity support).

Reads pre-cut 9000-sample waveforms from HDF5 files with companion
metadata CSVs. Supports multiple datasets (CEED, TW, STEAD, etc.).

H5 structure: /{category}/{split}/waveforms → (N, 3, 9000)
Metadata CSV columns: hdf5_index, sample_id, category, sampling_category,
    split, source_file, p_arrival_sample, s_arrival_sample, ps_diff_samples,
    [p_polarity]  ← optional: U=up, D=down, N=undetermined

When p_polarity column is absent or value is 'N'/NaN, polarity target is
all zeros → loss auto-masked via weight=|target|=0.

Config format (as in the "data" block of configs/train_rp90_motion.json):
    "data": {
        "CEED_NC": {"file_path": "DATA_ROOT/CEED_NC"},
        "CEED_SC": {"file_path": "DATA_ROOT/CEED_SC"},
        ...
    }
Each directory contains {dataset}_{category}[_sNNN].h5 + *_metadata.csv.
"""
from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DATA_LENGTH = 9000  # default, overrideable

# Polarity mapping
POL_MAP = {
    'U': '+', 'u': '+', '+': '+', 'positive': '+',
    'D': '-', 'd': '-', '-': '-', 'negative': '-',
    # 'N'/undetermined preserved explicitly — these are ambiguous (emergent/noisy)
    # P arrivals that the analyst couldn't polarity-label. Used as abstention
    # supervision: target=0 at P, but weight is nonzero → trains head toward 0.
    'N': 'N', 'n': 'N', 'undetermined': 'N', '?': 'N',
}


# ─────────────────────────────────────────────────────────────────────
# Label generation
# ─────────────────────────────────────────────────────────────────────
def gen_gaussian(T: int, center: int, mask_window: int) -> np.ndarray:
    target = np.zeros(T, dtype=np.float32)
    if center < 0 or center >= T:
        return target
    half = mask_window // 2
    sigma = max(half // 2, 1)
    gaus = np.exp(-(np.arange(-half, half + 1)) ** 2 / (2 * sigma ** 2))
    s = max(0, center - half)
    e = min(center + half + 1, T)
    gs = s - (center - half)
    target[s:e] = gaus[gs:gs + (e - s)]
    return target


def gen_picker_target(
    T: int, p: int, s: int, mw: int, suppress_pcoda: bool = False,
) -> np.ndarray:
    t = np.zeros((3, T), dtype=np.float32)
    if p >= 0: t[0] = gen_gaussian(T, p, mw)
    if s >= 0: t[1] = gen_gaussian(T, s, mw)
    t[2] = np.clip(1.0 - t[0] - t[1], 0, 1)
    if suppress_pcoda and p >= 0 and s > p:
        # Zero the noise channel in the P-to-S coda interval so the model
        # receives no gradient there for regional waveforms (CREW) where the
        # gap contains energetic P-coda, not ambient noise.
        #
        # IMPORTANT: this sets all three channels to 0 in the coda region
        # (P and S Gaussians have decayed to ~0 there), which breaks the
        # per-timestep sum-to-1 invariant.  The caller's loss function MUST
        # mask out (ignore gradient at) any timestep where all targets are
        # zero — i.e. use a masked cross-entropy or KL that skips zero rows.
        # If the loss is unmasked, the all-zero target will push all output
        # channels toward 0 and destabilise training.
        half = mw // 2
        # Use min(T, ...) on coda_start and min(T, max(0, ...)) on coda_end
        # so both bounds are always inside [0, T].  The guard below is still
        # needed for short PS diffs where s - p < mw (coda region collapses).
        coda_start = min(T, p + half)
        coda_end   = min(T, max(0, s - half))
        if coda_start < coda_end:
            t[2, coda_start:coda_end] = 0.0
    return t


def gen_picker_target_multi(T, p_list, s_list, mw):
    """Multi-peak picker target for EEWA/MMWA mosaic samples.

    Sums per-pick Gaussians on the P and S channels (clipped to [0, 1]),
    with the noise channel as the residual.
    """
    t = np.zeros((3, T), dtype=np.float32)
    for p in p_list:
        if 0 <= int(p) < T:
            t[0] = np.maximum(t[0], gen_gaussian(T, int(p), mw))
    for s in s_list:
        if 0 <= int(s) < T:
            t[1] = np.maximum(t[1], gen_gaussian(T, int(s), mw))
    t[2] = np.clip(1.0 - t[0] - t[1], 0, 1)
    return t


def gen_polarity_target(T, p, pol, mw):
    """1-channel signed polarity target with explicit loss weight.

    Returns:
        target: (1, T) float32 in [-1, +1].
                +Gaussian at P for up, -Gaussian for down, 0 for N/unlabeled.
        weight: (1, T) float32 in [0, +1].
                Gaussian at P for U / D / N (train head to match target at P),
                zero elsewhere. For unlabeled samples (no polarity info), all-zero.

    Three supervision regimes:
        U (up):   target=+Gaussian, weight=Gaussian  → head pushed toward +1 at P
        D (down): target=-Gaussian, weight=Gaussian  → head pushed toward -1 at P
        N (ambiguous): target=0,    weight=Gaussian  → head pushed toward  0 at P
                                                       (abstention — teaches the
                                                       head to recognize emergent/
                                                       noisy arrivals and not
                                                       predict confidently)
        anything else (unlabeled): target=0, weight=0 → no gradient
    """
    target = np.zeros((1, T), dtype=np.float32)
    weight = np.zeros((1, T), dtype=np.float32)
    if 0 <= p < T and pol in ('+', '-', 'N'):
        gauss = gen_gaussian(T, p, mw)
        weight[0] = gauss
        if pol == '+':
            target[0] = gauss
        elif pol == '-':
            target[0] = -gauss
        # pol == 'N': target stays zero; weight drives head toward 0 at P
    return target, weight


def gen_polarity_target_softmax_ce(T, p, pol, label_width, mask_width):
    """3-channel polarity target for masked softmax-CE loss (PhaseNet+ style,
    with N-active supervision).

    Uses separate `label_width` and `mask_width` (PhaseNet+ design):
      - `label_width` controls the Gaussian target's sharpness at P
      - `mask_width` controls how wide the loss window is around P
    The mask is BINARY (1 inside ±mask_width/2 around P, 0 elsewhere),
    giving uniform gradient weight across the supervised window — more
    stable and higher per-pick gradient volume than a Gaussian-shaped mask.

    Returns:
        target: (3, T) float32 — channels [N, U, D].
                At labeled P with U: target ≈ (0, 1, 0) at peak, decaying to (1, 0, 0) away
                At labeled P with D: target ≈ (0, 0, 1) at peak
                At labeled P with N: target = (1, 0, 0) at peak (active abstention)
                Elsewhere: target = (1, 0, 0)
        mask:   (1, T) float32 — 1.0 within ±mask_width/2 around P for U/D/N,
                zero for unlabeled samples and non-P timesteps.

    Difference from PhaseNet+:
      - Our mask is nonzero at N picks too (PhaseNet+ skips N entirely).
        Trains the head to actively output high N-class probability when
        the analyst couldn't determine polarity — the abstention signal.
    """
    target = np.zeros((3, T), dtype=np.float32)
    mask = np.zeros((1, T), dtype=np.float32)
    if 0 <= p < T and pol in ('+', '-', 'N'):
        gauss = gen_gaussian(T, p, label_width)
        half_m = mask_width // 2
        lo, hi = max(0, p - half_m), min(T, p + half_m)
        mask[0, lo:hi] = 1.0                      # BINARY loss window
        if pol == '+':
            target[1] = gauss                     # U channel
        elif pol == '-':
            target[2] = gauss                     # D channel
        # pol == 'N': target[1] and target[2] stay 0 → target[0] = 1 at P
    # Neutral class fills the remainder so the 3 channels sum to 1 per timestep
    target[0, :] = np.maximum(0, 1.0 - target[1, :] - target[2, :])
    return target, mask


def gen_event_center_target(T, p, s, mask_window=80):
    """1-channel sparse Gaussian peak at the midpoint of (P, S).

    Used by the v7 hybrid auxiliary detector head: a noise-robust gate that
    fires only when an actual event center is present. Sparse target → BCE
    drives output to 0 outside event centers (PhaseNet+ event_center style).

    Returns:
        target: (1, T) float32. Gaussian peak at midpoint(p, s) for EQ traces.
                All-zero for noise traces, missing-P, missing-S, or both.
    """
    target = np.zeros((1, T), dtype=np.float32)
    has_p = (0 <= p < T); has_s = (0 <= s < T)
    if has_p and has_s:
        center = int((p + s) // 2)
        if 0 <= center < T:
            target[0] = gen_gaussian(T, center, mask_window)
    return target


def gen_polarity_target_softmax_ud(T, p, pol, label_width, mask_width):
    """2-channel polarity target for masked softmax-CE on [U, D] only.

    Case B variant of softmax_ce: drops the N abstention class. The impulsive
    head independently provides emergent-vs-impulsive supervision, so the
    polarity head can focus solely on direction (U vs D) — direct cross-entropy
    on a clean binary direction class. Mask is nonzero ONLY at picks with U or
    D labels; N picks and unlabeled samples contribute zero loss to this head
    (impulsive head still supervises them).

    Returns:
        target: (2, T) float32 — channels [U, D].
                At labeled P with U: target ≈ (1, 0) at Gaussian peak
                At labeled P with D: target ≈ (0, 1) at peak
                Elsewhere (or N pick / no pick): target = (0, 0)  [unsupervised]
        mask:   (1, T) float32 — 1.0 within ±mask_width/2 around P for U or D
                ONLY. Zero at N picks and unlabeled samples.
    """
    target = np.zeros((2, T), dtype=np.float32)
    mask = np.zeros((1, T), dtype=np.float32)
    if 0 <= p < T and pol in ('+', '-'):
        gauss = gen_gaussian(T, p, label_width)
        half_m = mask_width // 2
        lo, hi = max(0, p - half_m), min(T, p + half_m)
        mask[0, lo:hi] = 1.0
        if pol == '+':
            target[0] = gauss   # U channel
        else:
            target[1] = gauss   # D channel
    return target, mask


def gen_impulsive_target(T, p, pol, mw, bg_weight=0.0):
    """1-channel impulsive/emergent target derived from polarity label.

    Interpretation:
        U or D polarity → impulsive  (analyst could determine first motion)
        N polarity      → emergent   (analyst couldn't determine first motion)
        no polarity col → no signal  (can't derive)

    bg_weight adds "be 0 everywhere" supervision on NOISE traces only:
      * bg_weight == 0.0  (default, backward-compatible):
            weight is non-zero ONLY in the ~±mw window around a LABELED pick
            (Gaussian). Off-pick → zero loss → the sigmoid floats at ~0.5 there.
      * bg_weight > 0.0  (e.g. 1.0):
            traces with NO P pick (pure noise, Sonly) get weight = bg_weight
            everywhere with target 0 → the head learns "quiet/noise Z → 0",
            which generalizes to the off-pick background of continuous data
            (kills the ~0.5–0.7 floor). EVENT traces are UNCHANGED — they keep
            the pick-window-only Gaussian weight, so the impulsive bump is not
            diluted by a competing "be 0" signal on its own trace. (An earlier
            version put bg_weight on event traces too; that swamped the ~50 bump
            samples under ~8950 "be 0" samples and the head collapsed to ≈0
            everywhere — DON'T do that.) Traces with a P pick but UNLABELED
            polarity (pol == '') still contribute nothing.

    Returns:
        target: (1, T) float32 in [0, +1]. Gaussian at P for U/D; zero elsewhere.
        weight: (1, T) float32 in [0, +1].
    """
    target = np.zeros((1, T), dtype=np.float32)
    has_pick = (0 <= p < T)
    labeled = has_pick and pol in ('+', '-', 'N')
    if labeled:
        # Event trace with a known polarity: supervise the pick window ONLY
        # (Gaussian) — exactly the legacy behavior. NO bg weight on this trace.
        gauss = gen_gaussian(T, p, mw)
        weight = np.zeros((1, T), dtype=np.float32)
        weight[0] = gauss
        if pol in ('+', '-'):
            target[0] = gauss                       # impulsive → Gaussian peak at 1
        # pol == 'N': target stays 0 at the pick → emergent supervision
    elif not has_pick:
        # No P pick (pure noise / Sonly): supervise impulsive → 0 everywhere.
        weight = np.full((1, T), bg_weight, dtype=np.float32)
        # target stays all-zero
    else:
        # P pick present but polarity unlabeled (pol == ''): contribute nothing.
        weight = np.zeros((1, T), dtype=np.float32)
    return target, weight


def gen_detector_target(T, p, s, mw=40):
    has_p, has_s = (0 <= p < T), (0 <= s < T)
    if not has_p and not has_s:
        t = np.zeros((2, T), dtype=np.float32); t[1] = 1.0; return t
    mask = np.zeros(T, dtype=np.float32)
    if has_p and has_s:
        if s < p: mask[:s] = 1.0; mask[p:] = 1.0
        else: mask[p:s + 1] = 1.0
    elif has_p: mask[p:] = 1.0
    elif has_s: mask[:s + 1] = 1.0
    t = np.zeros((2, T), dtype=np.float32)
    t[0] = np.clip(mask, 0, 1); t[1] = 1.0 - t[0]
    return t


def gen_detector_target_multi(T, p_list, s_list):
    """Multi-event detector mask for EEWA/MMWA mosaic samples.

    Picks are sorted, then orphans are peeled off the boundaries before
    pairing the remainder by index:

      - **Leading orphan S** (s_in[0] < p_in[0]) — event began before window.
        Mask covers [0 : s + 1] for each leading orphan S.
      - **Trailing orphan P** (p_in[-1] > s_in[-1]) — event continues past
        window. Mask covers [p : T] for each trailing orphan P.

    Examples:
        3P + 2S  (no orphan-S)         → (P0,S0) + (P1,S1) + extension [P2:T]
        1P + 2S  (lead-S, no trail-P)  → extension [0:S0+1] + (P0,S1)
        2P + 3S  (lead-S, no trail-P)  → extension [0:S0+1] + (P0,S1) + (P1,S2)
        2P + 1S  (no lead-S, trail-P)  → (P0,S0) + extension [P1:T]
        2P + 2S  with S0<P0<S1<P1     → extension [0:S0+1] + (P0,S1) + extension [P1:T]

    Picks outside [0, T) are silently dropped. Returns (2, T) one-hot target
    (channel 0 = event, channel 1 = no-event).
    """
    p_in = sorted(int(p) for p in p_list if 0 <= int(p) < T)
    s_in = sorted(int(s) for s in s_list if 0 <= int(s) < T)
    if not p_in and not s_in:
        t = np.zeros((2, T), dtype=np.float32); t[1] = 1.0; return t

    mask = np.zeros(T, dtype=np.float32)

    # Peel leading orphan S's: any S that comes before the earliest remaining P.
    while s_in and (not p_in or s_in[0] < p_in[0]):
        s = s_in.pop(0)
        mask[:s + 1] = 1.0

    # Peel trailing orphan P's: any P that comes after the latest remaining S.
    while p_in and (not s_in or p_in[-1] > s_in[-1]):
        p = p_in.pop()
        mask[p:] = 1.0

    # Remaining picks have len(p_in) == len(s_in) and proper temporal order;
    # pair-by-index produces one block per event. Use ``p <= s`` so a
    # zero-residual pair (P == S, very short P-S) still gets a one-sample
    # mask block rather than being silently dropped (which would label a
    # real event as pure noise).
    for p, s in zip(p_in, s_in):
        if p <= s:
            mask[p:s + 1] = 1.0

    t = np.zeros((2, T), dtype=np.float32)
    t[0] = np.clip(mask, 0, 1)
    t[1] = 1.0 - t[0]
    return t


def normalize_per_channel(wf):
    out = wf.copy()
    for c in range(out.shape[0]):
        std = out[c].std()
        out[c] = (out[c] - out[c].mean()) / std if std > 1e-6 else 0.0
    return out


def moving_normalize_per_channel(wf, filter_size=1024):
    """Per-channel sliding-window normalization (PhaseNet+ / EQNet style).

    For each channel and each timestep t, divide by the LOCAL absolute mean
    over a ±filter_size/2 window centred on t (subtracted by the LOCAL mean
    first). Zero-phase reflect-padding at the edges. This mirrors EQNet's
    `moving_normalize` (eqnet/models/unet.py:15) and is fundamentally
    different from global per-channel z-score:

      Global z-score: ONE big spike inflates the whole-window std → noise
        regions get amplified relative to true scale → detector fires often.

      Moving normalize: noise regions normalized by local noise std (~1),
        event regions normalized by local event std (~1) → both regions on
        the same scale → detector sees crisp ON/OFF boundary.

    PhaseNet+ uses filter=1024 samples = 10.24s @ 100 Hz. The window is small
    relative to our 9000-sample window (90s), so the moving filter captures
    short-term amplitude variability while preserving the relative event vs
    noise contrast.

    Args:
        wf: (3, T) float array.
        filter_size: moving window length in samples (default 1024 = 10.24s).

    Returns:
        (3, T) float32 normalized waveform.
    """
    from scipy.ndimage import uniform_filter1d
    out = wf.copy().astype(np.float32)
    # Reflect-padded moving mean per channel (axis=-1 = time)
    moving_mean = uniform_filter1d(out, size=filter_size, axis=-1, mode='reflect')
    out = out - moving_mean
    # Moving |x| as L1 std (matches EQNet); avoid div-by-zero
    moving_abs = uniform_filter1d(np.abs(out), size=filter_size, axis=-1, mode='reflect')
    moving_abs = np.where(moving_abs > 1e-8, moving_abs, 1.0)
    return (out / moving_abs).astype(np.float32)


def parse_arrival(val):
    """Parse p/s arrival sample from CSV (may be '[1234]' or '1234' or NaN).

    Returns the FIRST pick as int (or -1 on missing). Use for single-event
    categories (singleEQ, Ponly, Sonly, noise). For EEWA/MMWA mosaic samples
    use ``parse_arrival_list`` to retain every pick.

    Uses ``int(round(...))`` so fractional values (e.g. 5081.6) round to
    nearest integer instead of truncating downward — matches the rounding
    convention used by the data builder adapters when picks are written.
    """
    if pd.isna(val):
        return -1
    s = str(val).strip()
    if s.startswith('['):
        s = s.strip('[]')
    try:
        return int(round(float(s.split(',')[0])))  # take first if multi-peak
    except (ValueError, IndexError):
        return -1


def parse_arrival_list(val):
    """Parse p/s arrival sample list from CSV (multi-peak supported).

    Accepts '[1234, 5678]', '1234', or NaN. Returns a list of ints
    (empty list when the field is absent/unparseable).

    Uses ``int(round(...))`` so fractional values round to nearest integer
    rather than truncating — matches the adapter-side rounding convention
    and prevents systematic 1-sample label drift.
    """
    if pd.isna(val):
        return []
    s = str(val).strip()
    if s.startswith('['):
        s = s.strip('[]')
    if not s:
        return []
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(round(float(tok))))
        except (ValueError, IndexError):
            continue
    return out


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────
class MultiDatasetH5(Dataset):
    """
    Multi-dataset H5 reader for RED-PAN 90s with polarity support.

    Discovers H5+CSV pairs from data directories, builds a flat index,
    and samples per epoch with category weights.

    Args:
        data_dirs: list of dataset root dirs (each contains *.h5 + *_metadata.csv)
        split: 'train' or 'val'
        data_length: expected waveform length (9000)
        mask_window: Gaussian target half-width
        samples_per_epoch: how many samples to draw each epoch
        category_weights: {sampling_category: weight} for balanced sampling
        seed: random seed
    """

    def __init__(
        self,
        data_dirs: List[str],
        split: str = 'train',
        data_length: int = DATA_LENGTH,
        mask_window: int = 40,
        polarity_mask_window: Optional[int] = None,
        polarity_target_type: str = 'signed',  # 'signed' (1-ch tanh), 'softmax_ce' (3-ch [N,U,D]), or 'softmax_ud' (2-ch [U,D])
        polarity_label_width: int = 20,        # Gaussian σ-width of the target (softmax_ce only)
        polarity_mask_width: int = 30,         # BINARY loss-window width (softmax_ce only, PhaseNet+ style)
        impulsive_bg_weight: float = 0.0,      # >0 → full-trace supervision of the impulsive head (be 0 off-pick); 0 = pick-window only
        samples_per_epoch: int = 100000,
        category_weights: Optional[Dict[str, float]] = None,
        polarity_oversample: float = 1.0,       # >1.0 upweights polarity-bearing datasets within each category
        return_event_center: bool = False,      # v7 hybrid: append event_center target as 9th return element
        seed: int = 42,
        augment: bool = True,
        dataset_hp_freqs: Optional[Dict[str, float]] = None,  # {dataset_name: HP corner Hz}
        exclude_categories: Optional[List[str]] = None,  # categories to skip during _discover (e.g. ["singleEQ_zeropad"])
        dataset_categories: Optional[Dict[str, List[str]]] = None,  # {dataset_name: [allowed file_cats]} whitelist
        normalize_mode: str = 'zscore',  # 'zscore' (global) or 'moving' (PhaseNet+ style)
        moving_filter_size: int = 1024,  # samples; moving normalize window length
        pcoda_suppress_threshold: int = 0,  # samples; 0 = disabled.  Set to e.g. 1000
                                            # to zero out the picker noise channel in
                                            # the P-to-S interval for large-PS-diff
                                            # samples (regional EQ / CREW).
    ):
        self.data_length = data_length
        self.mask_window = mask_window
        # Polarity-specific target width for SIGNED mode. When None, reuse mask_window.
        # Narrower width concentrates the signed Gaussian → stronger per-peak
        # gradient drive for the tanh polarity head.
        self.polarity_mask_window = (
            mask_window if polarity_mask_window is None else polarity_mask_window)
        assert polarity_target_type in ('signed', 'softmax_ce', 'softmax_ud'), \
            f"polarity_target_type must be 'signed' / 'softmax_ce' / 'softmax_ud', got {polarity_target_type}"
        self.polarity_target_type = polarity_target_type
        # PhaseNet+ style label + mask widths for SOFTMAX_CE mode only.
        # label_width → Gaussian target sharpness; mask_width → binary loss window.
        self.polarity_label_width = polarity_label_width
        self.polarity_mask_width = polarity_mask_width
        self.impulsive_bg_weight = float(impulsive_bg_weight)
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch
        self.split = split
        self.polarity_oversample = float(polarity_oversample)
        self.return_event_center = return_event_center
        self.rng = random.Random(seed)
        self._h5_handles: Dict[str, h5py.File] = {}
        # Per-dataset highpass filter: applied to the waveform AND raw Z stream
        # (preserves first-motion polarity since HP is zero-phase Butterworth on
        # all 3 channels equally). Use for low-frequency-noise-dominated sources
        # like GeoNet (see audit: SNR_z 2.9 → 18.8 with HP=1).
        self.dataset_hp_freqs: Dict[str, float] = dataset_hp_freqs or {}
        # Lazy-build SOS filter coefs per (dataset, freq) on first use.
        self._hp_sos_cache: Dict[float, np.ndarray] = {}
        # Categories to skip entirely during _discover. Use to deprecate a
        # category without moving its H5 files (e.g. "singleEQ_zeropad").
        self.exclude_categories = set(exclude_categories or [])
        # Per-dataset category whitelist. Format: {ds_name: [allowed_file_cats]}.
        # Datasets not present in the dict → no filter (all file_cats allowed).
        # Datasets present but empty list → ALL files filtered out (effectively
        # excludes that dataset). Use case: train on CEED full + only "noise"
        # H5 files from other datasets, by setting:
        #   {"TW":["noise"], "INSTANCE":["noise"], ...}  (CEED_NC, CEED_SC absent)
        self.dataset_categories: Dict[str, List[str]] = dataset_categories or {}
        # Normalization mode for the picker/detector input branch.
        # 'zscore'  = global per-channel z-score (legacy, v3-v15)
        # 'moving'  = sliding-window per-time normalize (PhaseNet+ / EQNet style)
        # The polarity branch (z_raw) is unaffected — always uses raw scaled Z.
        if normalize_mode not in ('zscore', 'moving'):
            raise ValueError(f"normalize_mode must be 'zscore' or 'moving', got {normalize_mode!r}")
        self.normalize_mode = normalize_mode
        self.moving_filter_size = int(moving_filter_size)
        self.pcoda_suppress_threshold = int(pcoda_suppress_threshold)

        # Default category weights if not specified
        if category_weights is None:
            category_weights = {
                'singleEQ_00-05s': 0.10, 'singleEQ_05-10s': 0.10,
                'singleEQ_10-15s': 0.10, 'singleEQ_15-20s': 0.10,
                'singleEQ_20s_plus': 0.10,
                'noise': 0.15, 'Ponly': 0.08, 'Sonly': 0.03,
                'EEWA': 0.05, 'MMWA': 0.05, 'drop_noise': 0.02,
            }
        self.category_weights = category_weights

        # Build flat index: list of dicts
        self.index: Dict[str, List[dict]] = defaultdict(list)  # cat → entries
        self._discover(data_dirs)

        total = sum(len(v) for v in self.index.values())
        cats_str = ", ".join(f"{k}:{len(v)}" for k, v in sorted(self.index.items()))
        logger.info(f"MultiDatasetH5 {split}: {total} samples ({cats_str})")

        self.epoch_indices: List[dict] = []
        self._prepare_epoch(0)

    def _discover(self, data_dirs: List[str]):
        """Scan directories for H5+CSV pairs and build index."""
        for data_dir in data_dirs:
            if not os.path.isdir(data_dir):
                logger.warning(f"Skipping {data_dir}: not found")
                continue
            # Dataset name = parent directory basename (e.g. "TW", "GeoNet").
            ds_name = os.path.basename(os.path.normpath(data_dir))
            csvs = sorted(f for f in os.listdir(data_dir)
                          if f.endswith('_metadata.csv') and not f.endswith('.bak'))
            for csv_name in csvs:
                csv_path = os.path.join(data_dir, csv_name)
                # Corresponding H5: replace _metadata.csv with .h5
                h5_name = csv_name.replace('_metadata.csv', '.h5')
                h5_path = os.path.join(data_dir, h5_name)
                if not os.path.exists(h5_path):
                    continue

                # Skip excluded categories before parsing the (potentially huge) CSV.
                # Filename convention: <dataset>_dataset_90s_<category>_metadata.csv
                # Match the longest category suffix to avoid singleEQ matching
                # singleEQ_zeropad.
                base = csv_name[:-len("_metadata.csv")]
                file_cat = base.split("_dataset_90s_", 1)[-1] if "_dataset_90s_" in base else base
                if file_cat in self.exclude_categories:
                    logger.info(f"  excluding {ds_name}/{file_cat} (in exclude_categories)")
                    continue
                # Per-dataset whitelist filter: if this dataset is named in
                # dataset_categories, drop any file_cat not on its allowed list.
                # Datasets absent from the dict pass through unfiltered.
                # Case-insensitive match (consistent with the row-level filter
                # below that uses .lower()) — without this, a whitelist like
                # ["MOSAIC"] silently passes files named *_mosaic.h5 because
                # plain `not in` is case-sensitive.
                ds_allowed = self.dataset_categories.get(ds_name)
                if ds_allowed is not None:
                    ds_allowed_lower_set = {a.lower() for a in ds_allowed}
                    if file_cat.lower() not in ds_allowed_lower_set:
                        logger.info(
                            f"  filtering {ds_name}/{file_cat} (not in "
                            f"dataset_categories[{ds_name}]={ds_allowed})")
                        continue

                df = pd.read_csv(csv_path, low_memory=False)
                if 'split' in df.columns:
                    df = df[df['split'] == self.split]
                if df.empty:
                    continue

                # Accept both 'p_polarity' (CEED convention) and 'polarity' (TW
                # first-motion convention). Without this alias, the TW polarity
                # column name written by the TFRecord to H5 conversion is
                # silently ignored — every TW polarity sample loads with
                # has_polarity=False and the polarity head sees no TW signal.
                pol_col = (
                    'p_polarity' if 'p_polarity' in df.columns
                    else 'polarity' if 'polarity' in df.columns
                    else None
                )
                has_polarity = pol_col is not None

                # Pre-compute lowercased whitelist for row-level cat_root match.
                # When a dataset is restricted to e.g. ['noise'] at the file level,
                # individual H5 files (TW/noise.h5 in particular) sometimes still
                # contain rows with category='singleEQ/...' — i.e. mixed event
                # samples in a "noise" container. The file-level filter cannot
                # catch these; we additionally drop rows whose cat root isn't
                # on the whitelist when one is specified.
                ds_allowed_lower = (
                    {a.lower() for a in ds_allowed} if ds_allowed is not None
                    else None
                )

                for _, row in df.iterrows():
                    cat = str(row.get('category', 'unknown'))
                    # Row-level exclusion: covers cases where a "main" CSV
                    # contains rows with a sub-category like "singleEQ_zeropad/...".
                    # Match exact name OR any prefix-up-to-slash.
                    cat_root = cat.split('/', 1)[0]
                    if cat in self.exclude_categories or cat_root in self.exclude_categories:
                        continue
                    # Row-level whitelist: skip rows whose cat root isn't in the
                    # per-dataset allowed list (when one is set).
                    if ds_allowed_lower is not None and cat_root.lower() not in ds_allowed_lower:
                        continue
                    is_multi = cat in ('EEWA', 'MMWA')
                    if is_multi:
                        # Retain every pick — orphan-aware label generators
                        # (gen_detector_target_multi / gen_picker_target_multi)
                        # consume the full list.
                        p_list = parse_arrival_list(row.get('p_arrival_sample'))
                        s_list = parse_arrival_list(row.get('s_arrival_sample'))
                        p = p_list[0] if p_list else -1
                        s = s_list[0] if s_list else -1
                    else:
                        p_list = []
                        s_list = []
                        p = parse_arrival(row.get('p_arrival_sample'))
                        s = parse_arrival(row.get('s_arrival_sample'))

                    # Determine sampling category
                    samp_cat = str(row.get('sampling_category', cat))
                    cat_lower = cat.lower()
                    if 'drop' in cat_lower:
                        samp_cat = 'drop_noise'
                    elif 'singleEQ' in cat and 'zeropad' not in cat:
                        ps_diff = row.get('ps_diff_samples')
                        if pd.notna(ps_diff):
                            try:
                                dt = float(ps_diff) / 100.0
                                if dt < 5: samp_cat = 'singleEQ_00-05s'
                                elif dt < 10: samp_cat = 'singleEQ_05-10s'
                                elif dt < 15: samp_cat = 'singleEQ_10-15s'
                                elif dt < 20: samp_cat = 'singleEQ_15-20s'
                                else: samp_cat = 'singleEQ_20s_plus'
                            except (ValueError, TypeError):
                                pass

                    # Polarity — read from whichever column the source CSV uses.
                    if has_polarity:
                        raw_pol = str(row.get(pol_col, 'N'))
                        pol = POL_MAP.get(raw_pol.strip(), '')
                    else:
                        pol = ''
                    # Per-row polarity flag: TRUE only if this specific row has
                    # a usable polarity label (U/D/N), not just because the FILE
                    # has the polarity column. Otherwise polarity_oversample
                    # would inflate noise samples from CEED files (their pol=''
                    # but file-level has_polarity=True), diluting the actual
                    # polarity supervision signal.
                    has_polarity_row = (has_polarity and pol in ('+', '-', 'N'))

                    # Pure noise: no picks
                    is_noise = cat_lower in ('noise', 'noise/noise') and 'drop' not in cat_lower
                    if is_noise:
                        p, s = -1, -1

                    # Ponly / Sonly edge-position filter (v16 addition):
                    # Keep ONLY samples where the labeled pick lands within
                    # the first/last 5 seconds (500 samples @ 100 Hz) of the
                    # window. Semantics:
                    #   Ponly  = "event tail at end of window" → P ∈ [T-500, T)
                    #   Sonly  = "event head at start of window" → S ∈ [0, 500)
                    # Defensive: check BOTH samp_cat and cat substring — TW
                    # Ponly metadata occasionally has sampling_category mislabelled
                    # (eg 'singleEQ/Ponly_ps10_below' instead of 'Ponly') for ~10
                    # rows out of 51K, and we want to catch all of them.
                    EDGE_LEN = 500
                    is_ponly = (samp_cat == 'Ponly') or ('Ponly' in cat)
                    is_sonly = (samp_cat == 'Sonly') or ('Sonly' in cat)
                    if is_ponly and p >= 0 and p < self.data_length - EDGE_LEN:
                        continue  # drop this row (P too far inside)
                    if is_sonly and s >= 0 and s >= EDGE_LEN:
                        continue  # drop this row (S too far inside)

                    # H5 group path
                    group = str(row.get('storage_group', cat)).strip('/')

                    self.index[samp_cat].append({
                        'h5_path': h5_path,
                        'local_idx': int(row['hdf5_index']),
                        'group': group,
                        'p': p, 's': s,
                        'p_list': p_list, 's_list': s_list,
                        'category': cat,
                        'polarity': pol,
                        'sample_id': row.get('sample_id', ''),
                        'has_polarity': has_polarity_row,
                        'dataset_name': ds_name,
                    })

    def _prepare_epoch(self, epoch: int):
        self.rng.seed(epoch + 42)
        self.epoch_indices = []

        # Weighted sampling across categories
        total_weight = sum(self.category_weights.get(c, 0) for c in self.index)
        if total_weight < 1e-8:
            # Fallback: uniform
            all_entries = [e for entries in self.index.values() for e in entries]
            n = min(self.samples_per_epoch, len(all_entries))
            self.epoch_indices = self.rng.sample(all_entries, n)
            return

        oversample = self.polarity_oversample
        pol_count = 0
        for cat, entries in self.index.items():
            w = self.category_weights.get(cat, 0)
            if w <= 0 or not entries:
                continue
            n_cat = int(self.samples_per_epoch * w / total_weight)

            # Fast path: no oversample → original uniform sampling within category
            if oversample <= 1.0:
                drawn = self._draw_with_replacement(entries, n_cat)
                self.epoch_indices.extend(drawn)
                pol_count += sum(1 for e in drawn if e.get('has_polarity'))
                continue

            # Split the category into polarity-bearing vs non-polarity buckets.
            # Within-category draw is then allocated by an effective fraction
            # that upweights the polarity bucket by `oversample`.
            pol_entries = [e for e in entries if e.get('has_polarity')]
            nopol_entries = [e for e in entries if not e.get('has_polarity')]
            if not pol_entries or not nopol_entries:
                # Nothing to rebalance — fall back to uniform
                drawn = self._draw_with_replacement(entries, n_cat)
                self.epoch_indices.extend(drawn)
                pol_count += sum(1 for e in drawn if e.get('has_polarity'))
                continue

            p_frac = len(pol_entries) / len(entries)
            eff_frac = (p_frac * oversample) / (p_frac * oversample + (1 - p_frac))
            n_pol = int(round(n_cat * eff_frac))
            n_nopol = n_cat - n_pol
            self.epoch_indices.extend(self._draw_with_replacement(pol_entries, n_pol))
            self.epoch_indices.extend(self._draw_with_replacement(nopol_entries, n_nopol))
            pol_count += n_pol

        self.rng.shuffle(self.epoch_indices)
        total = len(self.epoch_indices)
        if total > 0:
            logger.info(
                f"MultiDatasetH5[{self.split}] epoch {epoch}: drew {total} samples "
                f"({pol_count} polarity-bearing, {100.0 * pol_count / total:.1f}%; "
                f"oversample={oversample:.1f})")

    def _draw_with_replacement(self, entries: List[dict], n: int) -> List[dict]:
        """Draw n entries from a pool, with replacement only if n > len(entries)."""
        if n <= 0 or not entries:
            return []
        if n <= len(entries):
            return self.rng.sample(entries, n)
        out: List[dict] = []
        out.extend(entries * (n // len(entries)))
        out.extend(self.rng.sample(entries, n % len(entries)))
        return out

    def set_epoch(self, epoch: int):
        self._prepare_epoch(epoch)

    def __len__(self):
        return len(self.epoch_indices)

    def _get_h5(self, path: str) -> h5py.File:
        if path not in self._h5_handles or not self._h5_handles[path].id.valid:
            self._h5_handles[path] = h5py.File(path, 'r')
        return self._h5_handles[path]

    def _apply_hp(self, wf_3T: np.ndarray, freq: float, sr: float = 100.0,
                  order: int = 4, taper_len: int = 200) -> np.ndarray:
        """Zero-phase Butterworth highpass per channel.

        Applies a Hann (cosine) taper of length `taper_len` samples at each
        end BEFORE filtering to suppress the edge ringing inherent in
        sosfiltfilt (which implicitly zero-pads the signal outside the window).
        Without the taper, large-amplitude DC/LF content at sample 0 or T-1
        produces a filter transient that can span ~3× filter-response samples
        (~1.8s for order-4 Butterworth at 1 Hz / 100 Hz fs), corrupting the
        signal exactly where Ponly/Sonly picks may sit. Default 200 samples
        (2 s) is sufficient for the 1 Hz Butterworth-4 transient envelope.
        """
        from scipy.signal import butter, sosfiltfilt
        if freq not in self._hp_sos_cache:
            self._hp_sos_cache[freq] = butter(order, freq, btype="highpass",
                                              fs=sr, output="sos")
        # Edge taper (per channel)
        n = wf_3T.shape[-1]
        if taper_len > 0 and 2 * taper_len < n:
            taper = np.ones(n, dtype=np.float32)
            cos_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper_len, dtype=np.float32) / taper_len))
            taper[:taper_len] = cos_in
            taper[-taper_len:] = cos_in[::-1]
            wf_3T = wf_3T * taper[None, :]
        return sosfiltfilt(self._hp_sos_cache[freq], wf_3T, axis=-1).astype(np.float32)

    def __getitem__(self, idx: int):
        entry = self.epoch_indices[idx]
        T = self.data_length

        try:
            hf = self._get_h5(entry['h5_path'])
            group = entry['group']
            # Navigate: try /{group}/{split}/waveforms or /{group}/waveforms
            for gp in [f"{group}/{self.split}/waveforms",
                       f"{group}/waveforms",
                       f"{self.split}/waveforms",
                       "waveforms"]:
                if gp in hf:
                    wf = hf[gp][entry['local_idx']].astype(np.float32)
                    break
            else:
                return self._fallback()
        except Exception:
            return self._fallback()

        # Ensure shape (3, T)
        if wf.shape[0] != 3:
            wf = wf.T
        if wf.shape[1] != T:
            if wf.shape[1] < T:
                wf = np.pad(wf, ((0, 0), (0, T - wf.shape[1])))
            else:
                wf = wf[:, :T]

        # NOTE on z_raw + HP ordering:
        # The polarity stream MUST see RAW Z (no HP, no normalize) to preserve
        # exact first-motion sign.
        # The picker/detector path benefits from per-dataset HP (TW/GeoNet)
        # to suppress long-period noise. So: save z_raw FIRST from the
        # untouched waveform, THEN apply HP to wf for the picker/detector
        # branch only.

        # Save RAW Z before any channel-destructive preprocessing. Polarity
        # branch needs the unfiltered waveform to preserve first-motion sign.
        z_raw = wf[2:3].copy().astype(np.float32)  # (1, T), channel index 2 = Z
        # Per-sample max-abs rescale — divides by a single positive scalar,
        # so sign and WITHIN-window relative amplitudes are preserved exactly.
        # Needed for fp16 AMP stability: raw counts can reach 10,000+; conv
        # activations then overflow fp16 (~65504 max). This is NOT the forbidden
        # z-scoring (no mean subtraction, no std division — both of which can
        # flip apparent first motion when pre-event baseline drifts).
        z_max = float(np.max(np.abs(z_raw)))
        if z_max > 1e-6:
            z_raw = z_raw / z_max

        # Per-dataset highpass for the picker/detector branch ONLY (z_raw is
        # already saved above and stays unfiltered). Skip HP for edge-of-window
        # pick categories (Ponly/Sonly) — even with the cosine taper, residual
        # edge artifacts in the first/last ~200 samples can land near the
        # labeled pick when it's at the edge.
        ds_name = entry.get('dataset_name', '')
        hp_freq = self.dataset_hp_freqs.get(ds_name, 0.0)
        cat = entry.get('category', '')
        skip_hp_categories = {'Ponly', 'Sonly'}
        if hp_freq > 0 and cat not in skip_hp_categories:
            wf = self._apply_hp(wf, hp_freq)

        if self.normalize_mode == 'moving':
            wf = moving_normalize_per_channel(wf, filter_size=self.moving_filter_size)
        else:
            wf = normalize_per_channel(wf)
        # Guard against pathological tiny-std channels producing |x|>>1 that
        # overflows fp16 downstream. Clip at 10 as a hard cap (loose bound,
        # affects neither valid z-score nor moving-normalize outputs typically).
        np.clip(wf, -10.0, 10.0, out=wf)

        p, s = entry['p'], entry['s']
        polarity_str = entry['polarity']

        # Augmentation: flip_polarity — negate all channels + flip polarity label
        # 50% chance, only for samples with polarity labels.
        # MUST flip z_raw consistently with wf so the polarity stream sees
        # the same direction of first motion as the label indicates.
        # Use self.rng (seeded) for reproducibility across runs/workers.
        if self.augment and polarity_str in ('+', '-') and self.rng.random() < 0.5:
            wf = -wf
            z_raw = -z_raw
            polarity_str = '-' if polarity_str == '+' else '+'

        is_multi = entry.get('category') in ('EEWA', 'MMWA')
        if is_multi:
            # Coda suppression is not applied to multi-event (EEWA/MMWA) samples:
            # they are mosaic composites, not regional single-event waveforms.
            picker = gen_picker_target_multi(
                T, entry.get('p_list', []), entry.get('s_list', []),
                self.mask_window,
            )
        else:
            # '>=' is intentional: threshold means "suppress for PS diff of AT
            # LEAST this many samples".  p >= 0 guard handles noise samples
            # (p == -1) and ensures s > p before computing the difference.
            suppress_pcoda = (
                self.pcoda_suppress_threshold > 0
                and p >= 0 and s > p
                and (s - p) >= self.pcoda_suppress_threshold
            )
            picker = gen_picker_target(T, p, s, self.mask_window,
                                       suppress_pcoda=suppress_pcoda)
        if self.polarity_target_type == 'softmax_ce':
            # PhaseNet+ style: separate label Gaussian width from binary mask width.
            pol, pol_weight = gen_polarity_target_softmax_ce(
                T, p, polarity_str,
                label_width=self.polarity_label_width,
                mask_width=self.polarity_mask_width,
            )
            # pol shape: (3, T), pol_weight shape: (1, T) binary mask for CE
        elif self.polarity_target_type == 'softmax_ud':
            # Case B: 2-channel [U, D] softmax — abstention handled by impulsive head.
            pol, pol_weight = gen_polarity_target_softmax_ud(
                T, p, polarity_str,
                label_width=self.polarity_label_width,
                mask_width=self.polarity_mask_width,
            )
            # pol shape: (2, T), pol_weight shape: (1, T) binary mask, only for U/D
        else:
            pol, pol_weight = gen_polarity_target(
                T, p, polarity_str, self.polarity_mask_window)
            # pol shape: (1, T) signed, pol_weight shape: (1, T) |target|-based
        imp, imp_weight = gen_impulsive_target(
            T, p, polarity_str, self.polarity_mask_window, bg_weight=self.impulsive_bg_weight)
        if is_multi:
            detector = gen_detector_target_multi(
                T, entry.get('p_list', []), entry.get('s_list', []),
            )
        else:
            detector = gen_detector_target(T, p, s, self.mask_window)

        out = [
            torch.from_numpy(wf),
            torch.from_numpy(z_raw),
            torch.from_numpy(picker),
            torch.from_numpy(pol),
            torch.from_numpy(pol_weight),
            torch.from_numpy(imp),
            torch.from_numpy(imp_weight),
            torch.from_numpy(detector),
        ]
        if self.return_event_center:
            # v7 hybrid: sparse Gaussian peak at midpoint(P, S) for noise-robust
            # event detection auxiliary head. All-zero on noise / unlabeled / one-of.
            event_center = gen_event_center_target(T, p, s, mask_window=80)
            out.append(torch.from_numpy(event_center))
        return tuple(out)

    def _fallback(self):
        T = self.data_length
        wf = np.random.randn(3, T).astype(np.float32) * 0.01
        z_raw = np.random.randn(1, T).astype(np.float32) * 0.01
        picker = np.zeros((3, T), dtype=np.float32); picker[2] = 1.0
        if self.polarity_target_type == 'softmax_ce':
            # 3-channel [N,U,D]; neutral class = 1 when no pick
            pol = np.zeros((3, T), dtype=np.float32); pol[0] = 1.0
        elif self.polarity_target_type == 'softmax_ud':
            # 2-channel [U,D]; both zero (no signal — pol_weight will be zero too)
            pol = np.zeros((2, T), dtype=np.float32)
        else:
            pol = np.zeros((1, T), dtype=np.float32)
        pol_weight = np.zeros((1, T), dtype=np.float32)
        imp = np.zeros((1, T), dtype=np.float32)
        # treat a fallback like a noise trace: supervise impulsive→0 everywhere if full-trace supervision is on
        imp_weight = np.full((1, T), self.impulsive_bg_weight, dtype=np.float32)
        det = np.zeros((2, T), dtype=np.float32); det[1] = 1.0
        out = [torch.from_numpy(wf), torch.from_numpy(z_raw),
               torch.from_numpy(picker),
               torch.from_numpy(pol), torch.from_numpy(pol_weight),
               torch.from_numpy(imp), torch.from_numpy(imp_weight),
               torch.from_numpy(det)]
        if self.return_event_center:
            out.append(torch.zeros((1, T), dtype=torch.float32))
        return tuple(out)

    def close(self):
        for h in self._h5_handles.values():
            h.close()
        self._h5_handles.clear()

    def __del__(self):
        self.close()
