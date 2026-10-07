"""Schema and validation for unified H5 builder.

Standard waveform shape: (3, 9000) channel-first, 100 Hz, channels (E, N, Z).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal
import numpy as np


T_SAMPLES = 9000          # 90 s @ 100 Hz
N_CHANNELS = 3            # (E, N, Z) channel-first
SAMPLING_RATE_HZ = 100
DT_SEC = 1.0 / SAMPLING_RATE_HZ

Category = Literal[
    "singleEQ", "singleEQ_zeropad", "Ponly", "Sonly",
    "EEWA", "MMWA", "noise",
]
# Polarity dictionary:
#   "U" / "D" — first-motion direction (analyst determined it)
#   "N"       — analyst examined the pick but found NO clear first motion
#               ("emergent"). Kept DISTINCT from "" so downstream the impulsive
#               head can be supervised toward 0 on N picks (and the [N,U,D]
#               softmax head's N channel gets real supervision) instead of N
#               picks being conflated with "no annotation at all".
#   ""        — no polarity annotation for this pick (don't supervise polarity).
Polarity = Literal["U", "D", "N", ""]
Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class WaveformSample:
    """One pre-padded sample ready for H5 storage.

    Adapter responsibility: produce this from native source format
    (mseed / sac / STEAD-hdf5 / INSTANCE-hdf5 / ...).

    Writer responsibility: validate the contract via __post_init__ then persist.

    Required:
        sample_id, waveform, category, split, source_file
    Optional:
        p_arrival_samples, s_arrival_samples, polarity, sampling_category,
        content_hash (set by writer if blank)

    Channel convention: (E, N, Z), so wf[2] = Z (used by polarity branch).
    Sampling convention: 100 Hz. Resampling, if needed, is the adapter's job.
    Bandpass convention: applied by the adapter. The writer is opaque to filtering.
    """
    sample_id: str
    waveform: np.ndarray
    category: Category
    split: Split = "train"
    source_file: str = ""
    p_arrival_samples: List[int] = field(default_factory=list)
    s_arrival_samples: List[int] = field(default_factory=list)
    polarity: Polarity = ""
    sampling_category: str = ""
    content_hash: str = ""

    def __post_init__(self):
        # Shape and dtype
        if self.waveform.shape != (N_CHANNELS, T_SAMPLES):
            raise ValueError(
                f"waveform must be ({N_CHANNELS}, {T_SAMPLES}); "
                f"got {self.waveform.shape}"
            )
        if self.waveform.dtype != np.float32:
            raise ValueError(f"waveform dtype must be float32; got {self.waveform.dtype}")
        if not np.isfinite(self.waveform).all():
            raise ValueError(f"waveform contains NaN/Inf in sample {self.sample_id}")

        # Pick bounds
        for p in self.p_arrival_samples:
            if not 0 <= p < T_SAMPLES:
                raise ValueError(f"p={p} out of [0, {T_SAMPLES}) in {self.sample_id}")
        for s in self.s_arrival_samples:
            if not 0 <= s < T_SAMPLES:
                raise ValueError(f"s={s} out of [0, {T_SAMPLES}) in {self.sample_id}")

        # Category-specific contract
        if self.category == "noise":
            if self.p_arrival_samples or self.s_arrival_samples:
                raise ValueError(
                    f"noise sample {self.sample_id} must have empty p/s arrivals"
                )
            if self.polarity:
                raise ValueError(f"noise sample {self.sample_id} cannot have polarity")
        elif self.category == "Ponly":
            if not self.p_arrival_samples or self.s_arrival_samples:
                raise ValueError(
                    f"Ponly sample {self.sample_id} must have p but not s"
                )
        elif self.category == "Sonly":
            if self.p_arrival_samples or not self.s_arrival_samples:
                raise ValueError(
                    f"Sonly sample {self.sample_id} must have s but not p"
                )
        elif self.category == "singleEQ" or self.category == "singleEQ_zeropad":
            if not self.p_arrival_samples or not self.s_arrival_samples:
                raise ValueError(
                    f"{self.category} {self.sample_id} must have both p and s"
                )
            if len(self.p_arrival_samples) != len(self.s_arrival_samples):
                raise ValueError(
                    f"{self.category} {self.sample_id}: p/s pick count mismatch"
                )
        elif self.category in ("EEWA", "MMWA"):
            # Multi-event mosaic: imbalanced p/s lists are EXPECTED — they
            # encode legitimate window-edge truncation. The dataset loader's
            # ``gen_detector_target_multi`` peels orphan picks (lead-S, trail-P)
            # before pair-by-index, so e.g. 3P+2S = paired(P0,S0) + paired(P1,S1)
            # + extension [P2:T]. The only invariant we enforce here is that
            # at least one pick survived in-window filtering.
            if not self.p_arrival_samples and not self.s_arrival_samples:
                raise ValueError(
                    f"{self.category} {self.sample_id}: no in-window picks; "
                    "would mislabel a real signal as noise"
                )

        # Polarity dictionary
        if self.polarity and self.polarity not in ("U", "D", "N"):
            raise ValueError(
                f"polarity must be 'U', 'D', 'N', or '' (got {self.polarity!r}) "
                f"in {self.sample_id}"
            )
        if self.polarity and self.category not in ("singleEQ", "Ponly"):
            raise ValueError(
                f"polarity only allowed on singleEQ/Ponly; got {self.category} "
                f"in {self.sample_id}"
            )

    @property
    def ps_diff_samples(self) -> int | None:
        """First-pair P-S difference in samples; None if either is missing."""
        if self.p_arrival_samples and self.s_arrival_samples:
            return self.s_arrival_samples[0] - self.p_arrival_samples[0]
        return None
