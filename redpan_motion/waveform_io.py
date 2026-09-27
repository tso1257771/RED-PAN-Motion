"""Waveform I/O and model-input preprocessing for redpan_motion.

SAC/StationXML reading, sensitivity correction, model-input bandpass, and
archive grouping.

    load(data_glob, xml=None)        -> raw_counts, wf_sensonly, inv, ids, t0, stream
    bandpass_for_model(raw, dt)      -> (3, T) float32 model input
    apply_sensitivity(stream, inv)   -> divide each trace by its sensitivity
    group_instruments(datadir, y, j, min_comps=3) -> {(net,sta,loc,chn2): glob_index}
"""
import glob
import logging
import os
from collections import defaultdict

import numpy as np
import obspy

from redpan_motion.response import CHN_FALLBACKS

# Moved to redpan_motion.signal so the filters carry no response dependency.
# Re-exported here because callers and the README use these names.
from redpan_motion.signal import bandpass_for_model, highpass  # noqa: F401

logger = logging.getLogger(__name__)

# channel last char -> component index; the vertical (Z/3) must land at index 2
ENZ_ORDER = {"E": 0, "1": 0, "N": 1, "2": 1, "Z": 2, "3": 2}
CH_Z = 2   # vertical row in the (3, T) arrays load() returns (matches the model's CH_Z)


def apply_sensitivity(st, inv):
    """Divide each trace by its instrument sensitivity (counts -> physical SI),
    with channel-code (HHN->HH1) and location ('', '--') fallbacks. Matches the
    sensitivity correction of the original RED-PAN without the TF dependency.
    Returns True iff every trace was corrected."""
    ok = True
    for tr in st:
        net, sta, loc, chn = (tr.stats.network, tr.stats.station,
                              tr.stats.location, tr.stats.channel)
        sens = None
        for c_try in CHN_FALLBACKS.get(chn, [chn]):
            for loc_try in [loc] + [lc for lc in ("", "--") if lc != loc]:
                for n in inv.select(network=net, station=sta,
                                    location=loc_try, channel=c_try):
                    for s in n:
                        for c in s:
                            if c.response and c.response.instrument_sensitivity:
                                sens = c.response.instrument_sensitivity.value
                                break
                        if sens:
                            break
                    if sens:
                        break
                if sens:
                    break
            if sens:
                break
        if sens and np.isfinite(sens) and sens != 0:
            tr.data = tr.data.astype(np.float64) / sens
        else:
            ok = False
    return ok




def load(data_glob, xml=None):
    """Load a 3-component instrument (+ inventory if ``xml`` given).

    Returns ``(raw_counts, wf_sensonly, inv, (net, sta, loc, chn_pre), t0,
    stream)``. ``raw_counts`` is (3, T) E/N/Z float64 unfiltered counts (for
    amplitude/response work); with ``xml=None`` the inventory + sensitivity-only
    waveform are skipped entirely (``wf_sensonly`` / ``inv`` are None) — the
    no-amplitude fast path."""
    st = obspy.read(data_glob)
    st.merge(method=1, fill_value=0)
    tr0 = st[0]
    net, sta, loc = tr0.stats.network, tr0.stats.station, tr0.stats.location
    chn_pre = tr0.stats.channel[:2]
    npts = min(len(tr.data) for tr in st)
    raw = np.zeros((3, npts), np.float64)
    for tr in st:
        k = ENZ_ORDER.get(tr.stats.channel[-1].upper())
        if k is not None:
            raw[k] = tr.data[:npts].astype(np.float64)
    if xml is None:                       # no-amplitude path: skip inventory + sensonly
        return raw, None, None, (net, sta, loc, chn_pre), tr0.stats.starttime, st
    inv = obspy.read_inventory(xml)
    # wf_sensonly: demean + bandpass 3-45 Hz + sensitivity correction (reamp
    # convention; bandpass keeps SNR from collapsing on low-frequency noise power)
    sens = st.copy().detrend("demean").filter("bandpass", freqmin=3.0, freqmax=45.0)
    if not apply_sensitivity(sens, inv):
        logger.warning("sensitivity correction incomplete for some channels")
    wf_sensonly = np.zeros((3, npts), np.float64)
    for tr in sens:
        k = ENZ_ORDER.get(tr.stats.channel[-1].upper())
        if k is not None:
            wf_sensonly[k] = tr.data[:npts].astype(np.float64)
    return raw, wf_sensonly, inv, (net, sta, loc, chn_pre), tr0.stats.starttime, st


def group_instruments(datadir, year, jday, min_comps=3):
    """Glob a day's files in a CWA-style day archive and group into instruments.
    Returns ``{(net, sta, loc, chn2): glob_index}`` for every instrument with at
    least ``min_comps`` seismometer (H/L/N) components. Default 3 = complete 3-C
    instruments only (safe for callers like daily_inference that don't pad gaps);
    ``min_comps=1`` also takes 1- and 2-component instruments, for a caller that
    fills the missing components itself.

    Filenames follow the day-archive convention ``STA.NET.LOC.CHN.YYYY.JJJ``;
    the returned glob_index uses a ``?`` wildcard on the component letter so
    ``obspy.read`` picks up the 3 traces in one call."""
    year_s, jday_s = f"{int(year):04d}", f"{int(jday):03d}"
    groups = defaultdict(set)
    for f in glob.glob(f"{datadir}/*/A/{year_s}/{jday_s}/*/*"):
        parts = os.path.basename(f).split(".")
        if len(parts) < 6:
            continue
        sta, net, loc, chn = parts[0], parts[1], parts[2], parts[3]
        if len(chn) < 3:
            continue
        comp = chn[-1].upper()
        # seismometer channels only: instrument code (2nd char) must be H/L/N
        # (high-gain, low-gain, accelerometer) AND a seismic orientation E/N/Z/1/2/3.
        # Excludes non-seismometers like BD/HD (pressure), BK, HA, etc.
        if chn[1].upper() not in ("H", "L", "N") or comp not in ENZ_ORDER:
            continue
        groups[(net, sta, loc, chn[:2], os.path.dirname(f))].add(comp)
    out = {}
    for (net, sta, loc, chn2, dirpath), comps in sorted(groups.items()):
        if len(comps) >= min_comps:                  # default 3 = complete instruments only
            key = (net, sta, loc, chn2)
            if key in out:                           # same instrument under two subtrees
                logger.warning("duplicate instrument %s in multiple dirs; using %r",
                               ".".join(key), dirpath)
            out[key] = f"{dirpath}/{sta}.{net}.{loc}.{chn2}?.{year_s}.{jday_s}"
    return out
