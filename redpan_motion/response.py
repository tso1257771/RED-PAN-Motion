"""Instrument response and magnitude-window constants.

Abbreviations used in this module and ``redpan_motion.amplitudes``:

- **WA** — Wood-Anderson (torsion seismometer simulation for local magnitude)
- **PAZ** — Poles And Zeros (instrument response representation)

Constants and simple helpers; numerical amplitude logic lives in
``redpan_motion.amplitudes``.
"""
from __future__ import annotations

import numpy as np
from obspy import Stream


# -----------------------------------------------------------------------------
# Wood-Anderson PAZ variants dispatched on input signal type.
# Standard WA: w0 = 7.854 rad/s (T0 = 0.8 s), h = 0.8, V0 = 2080.
# Matches the Wood-Anderson response used by the original RED-PAN.
# -----------------------------------------------------------------------------
WA_PAZ_VEL = {
    'sensitivity': 2080.0,
    'zeros': [0j],
    'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j],
    'gain': 1.0,
}
WA_PAZ_ACC = {
    'sensitivity': 2080.0,
    'zeros': [],
    'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j],
    'gain': 1.0,
}

VEL_CHN_SET = {'HH', 'BH', 'EH'}
ACC_CHN_SET = {'HN', 'HL'}

# -----------------------------------------------------------------------------
# Wadati-proxy magnitude window constants
# -----------------------------------------------------------------------------
WADATI_VPVS_FACTOR    = 8.0    # km per (ts-tp) second
MAG_WINDOW_TMIN_RATIO = 1.1    # post-S min window factor
MAG_WINDOW_ALPHA      = 0.1    # s/km distance-dependent tail
MAG_WINDOW_TMAX       = 80.0   # s absolute cap
MAG_WINDOW_PRE_P      = 1.0    # s pre-P buffer
WA_MARGIN_SEC         = 60.0   # s padding for filter edge effects

FS = 100.0  # target sample rate

# -----------------------------------------------------------------------------
# Channel name fallbacks: SAC files often label horizontals as HHN/HHE even
# when the StationXML registers them as HH1/HH2 (azimuth-deviated).
# Order = priority.
# -----------------------------------------------------------------------------
CHN_FALLBACKS = {
    'HHN': ['HHN', 'HH1'], 'HHE': ['HHE', 'HH2'],
    'BHN': ['BHN', 'BH1'], 'BHE': ['BHE', 'BH2'],
    'EHN': ['EHN', 'EH1'], 'EHE': ['EHE', 'EH2'],
    'HLN': ['HLN', 'HL1'], 'HLE': ['HLE', 'HL2'],
    'HNN': ['HNN', 'HN1'], 'HNE': ['HNE', 'HN2'],
}


def stream_to_enz_array(wf: Stream) -> np.ndarray:
    """Convert a 3-trace ObsPy Stream to (3, T) float64 in (E, N, Z) order."""
    _order = {'E': 0, '1': 0, 'N': 1, '2': 1, 'Z': 2, '3': 2}
    npts = min(len(tr.data) for tr in wf)
    arr = np.zeros((3, npts), dtype=np.float64)
    for tr in wf:
        idx = _order.get(tr.stats.channel[-1].upper())
        if idx is not None:
            arr[idx] = tr.data[:npts].astype(np.float64)
    return arr


def sensor_type(station_id: str) -> str:
    """Return 'velocity', 'acceleration', or 'unknown' from SEED channel prefix.

    station_id format: "NET.STA.LOC.CHN" where CHN is the 2-char band+instrument
    code (e.g. "TW.ZUZH.10.HN").
    """
    chn_prefix = station_id.split('.')[-1].upper()
    if chn_prefix in ('HL', 'HN', 'HG', 'LN', 'LG', 'BN', 'SN'):
        return 'acceleration'
    elif chn_prefix in ('EH', 'HH', 'BH', 'SH', 'LH', 'VH', 'UH', 'HF'):
        return 'velocity'
    else:
        return 'unknown'
