"""Model-input filters, with no dependency on ObsPy response handling.

``waveform_io`` also re-exports both names, so ``from redpan_motion.waveform_io
import highpass`` keeps working. They live here so that the SeisBench
integration can import a filter without pulling in the response and amplitude
layers, which it never uses.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_for_model(
    raw: np.ndarray, dt: float = 0.01, fmin: float = 3.0, fmax: float = 45.0
) -> np.ndarray:
    """Demean + 4th-order Butterworth bandpass each channel of the raw counts, an
    alternative model input to ``highpass``. (3, T) -> (3, T)."""
    nyq = 0.5 / dt
    sos = butter(4, [fmin / nyq, fmax / nyq], btype="band", output="sos")
    out = np.empty(raw.shape, dtype=np.float32)
    for c in range(raw.shape[0]):
        out[c] = sosfiltfilt(sos, raw[c] - raw[c].mean()).astype(np.float32)
    return out


def highpass(raw: np.ndarray, freq: float = 1.0, dt: float = 0.01) -> np.ndarray:
    """Demean + 4th-order zero-phase Butterworth highpass per channel, the
    recommended model input. 1 Hz matches how the redpan_motion checkpoint was
    trained: its TW and GeoNet data were high-passed at 1 Hz, the rest left raw.
    (3, T) -> (3, T). Call it explicitly before predict_arrays
    (read -> highpass -> model)."""
    nyq = 0.5 / dt
    sos = butter(4, freq / nyq, btype="high", output="sos")
    out = np.empty(raw.shape, dtype=np.float32)
    for c in range(raw.shape[0]):
        out[c] = sosfiltfilt(sos, raw[c] - raw[c].mean()).astype(np.float32)
    return out
