"""Physical-unit amplitude computation and SNR (signal-to-noise ratio) helpers.

``sensonly`` / ``_so`` suffixes denote sensitivity-only correction (divide by
instrument gain — no full spectral deconvolution). See ``response.py`` for WA
(Wood-Anderson) and PAZ (Poles And Zeros) definitions.

Two amplitude pathways are provided:

  * `window_amplitudes` - fast, sensitivity-only corrected traces.
    Applies a 0.5 Hz high-pass before integration to suppress DC drift.
    Output columns: amp_disp_nm_sensonly, amp_vel_nm_s_sensonly,
                    amp_acc_gal_sensonly.

  * `full_response_amplitudes` - canonical, full ObsPy response removal
    with spectral integration for accelerometers, then Wood-Anderson
    simulation. Output columns: amp_disp_nm, amp_vel_nm_s, amp_acc_gal,
    WA_amp_disp_mm.
"""
from __future__ import annotations

import numpy as np
from obspy import Stream, Trace
from scipy.ndimage import uniform_filter1d

from .response import (
    ACC_CHN_SET,
    CHN_FALLBACKS,
    FS,
    VEL_CHN_SET,
    WA_MARGIN_SEC,
    WA_PAZ_ACC,
    WA_PAZ_VEL,
)


# -----------------------------------------------------------------------------
# Sensitivity-only fast amplitude path
# -----------------------------------------------------------------------------

def window_amplitudes(wf_np, idx, win_npts, sensor_type, dt):
    """
    Compute peak displacement (nm), velocity (nm/s), and acceleration (gal)
    within a pick window, converting from raw SI physical units.

    NOTE: This function operates on sensitivity-only corrected traces
    (no full response removal). For accelerometer sensors, naive
    cumulative integration `np.cumsum(acc)` accumulates DC drift that
    dominates the true velocity signal (observed 60-120x bias vs
    co-located velocity sensors). We apply a high-pass filter
    (0.5 Hz, 4th-order zero-phase) before integrating to suppress this
    drift. For velocity sensors, the same high-pass is applied to vel
    before integrating to displacement, making disp estimates stable.
    """
    end = min(idx + win_npts, wf_np.shape[1])
    if end - idx < 4:  # need enough samples for filter
        nan = float('nan')
        return nan, nan, nan

    win = wf_np[:, idx:end].astype(np.float64)

    def _hp_integrate(x, dt, fc=0.5, order=4):
        """High-pass then cumulative-trapezoid integrate (stable)."""
        from scipy.signal import butter, sosfiltfilt
        fs = 1.0 / dt
        # Guard against too-short windows (filtfilt needs ~3*order*2 samples).
        if x.shape[-1] < 3 * order * 2 + 1:
            return np.cumsum(x - x[:, :1], axis=1) * dt
        sos = butter(order, fc, btype='highpass', fs=fs, output='sos')
        x_hp = sosfiltfilt(sos, x, axis=-1)
        # Cumulative trapezoidal rule in-axis
        return np.concatenate(
            [np.zeros_like(x_hp[:, :1]),
             np.cumsum(0.5 * (x_hp[:, :-1] + x_hp[:, 1:]), axis=1) * dt],
            axis=1,
        )

    if sensor_type == 'velocity':
        vel  = win
        disp = _hp_integrate(vel, dt)
        acc  = np.gradient(vel, dt, axis=1)
    elif sensor_type == 'acceleration':
        acc  = win
        vel  = _hp_integrate(acc, dt)
        disp = _hp_integrate(vel, dt)
    else:
        nan = float('nan')
        return nan, nan, nan

    amp_disp_nm  = float(np.max(np.abs(disp))) * 1e9
    amp_vel_nm_s = float(np.max(np.abs(vel)))  * 1e9
    amp_acc_gal  = float(np.max(np.abs(acc)))  * 100.0

    return amp_disp_nm, amp_vel_nm_s, amp_acc_gal


# -----------------------------------------------------------------------------
# SNR
# -----------------------------------------------------------------------------

def compute_snr(wf_np, idx, win_npts, noise_win_npts, noise_anchor=None):
    """RMS SNR.  noise_anchor defaults to idx (pre-signal noise window).
    For S picks pass noise_anchor=p_idx so we measure noise before the P
    arrival, not the P-wave coda which would inflate noise toward 1."""
    anchor = idx if noise_anchor is None else noise_anchor
    noise_start = max(0, anchor - noise_win_npts)
    noise_end   = anchor
    if noise_end <= noise_start:
        return float('nan')

    noise   = wf_np[:, noise_start:noise_end]
    sig_end = min(idx + win_npts, wf_np.shape[1])
    signal  = wf_np[:, idx:sig_end]

    noise_rms  = float(np.sqrt(np.mean(noise  ** 2)))
    signal_rms = float(np.sqrt(np.mean(signal ** 2)))

    if noise_rms <= 0:
        return float('nan')
    return signal_rms / noise_rms


# -----------------------------------------------------------------------------
# Adaptive S-coda amplitude window
# -----------------------------------------------------------------------------

def adaptive_s_window(wf_np, s_idx, nominal_npts, p_idx, dt,
                      smooth_sec=1.0, min_coda_sec=2.0,
                      min_coda_factor=0.3, spike_ratio=3.0,
                      secondary_ratio=2.0):
    """Adaptively shorten the S-coda amplitude window to avoid noise spikes."""
    total_npts = wf_np.shape[1]
    end_abs = min(s_idx + nominal_npts, total_npts)
    actual_nominal = end_abs - s_idx
    if actual_nominal < 2:
        return max(1, actual_nominal)

    sp_diff = s_idx - p_idx
    earliest = min(
        max(int(min_coda_sec / dt), int(min_coda_factor * sp_diff)),
        actual_nominal,
    )

    p_start = max(0, p_idx)
    variances = [np.var(wf_np[c, s_idx:end_abs]) for c in range(wf_np.shape[0])]
    best_c = int(np.argmax(variances))
    if variances[best_c] == 0:
        return actual_nominal

    seg = wf_np[best_c, p_start:end_abs].astype(np.float64)
    smooth_npts = max(3, int(smooth_sec / dt))
    envelope = uniform_filter1d(np.abs(seg), size=smooth_npts)

    s_rel = s_idx - p_start
    coda_end_rel = s_rel + min(int(10.0 / dt), sp_diff)
    coda_end_rel = min(coda_end_rel, len(envelope))

    main_peak = np.max(envelope[:coda_end_rel]) if coda_end_rel > 0 else 1e-30

    global_peak_rel = int(np.argmax(envelope))
    global_peak = envelope[global_peak_rel]
    if (global_peak_rel > coda_end_rel
            and global_peak > main_peak * secondary_ratio):
        valley_region = envelope[coda_end_rel:global_peak_rel]
        if len(valley_region) > 0:
            valley_rel = coda_end_rel + int(np.argmin(valley_region))
            trunc = max(valley_rel - s_rel, earliest)
            return min(trunc, actual_nominal)

    scan_start = s_rel + earliest
    scan_end = min(s_rel + actual_nominal, len(envelope))
    if scan_end > scan_start + 1:
        region = envelope[scan_start:scan_end]
        running_min = np.minimum.accumulate(region)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = region / (running_min + 1e-30)
        below_half = running_min < main_peak * 0.5
        spike_mask = (ratio > spike_ratio) & below_half
        if np.any(spike_mask):
            first_spike = int(np.argmax(spike_mask))
            min_before = int(np.argmin(region[:first_spike + 1]))
            trunc = max(scan_start + min_before - s_rel, earliest)
            return min(trunc, actual_nominal)

    return actual_nominal


# -----------------------------------------------------------------------------
# Full-response amplitude path (Wood-Anderson + physical units)
# -----------------------------------------------------------------------------

def phys_amps_in_window(phys_trcs, rel_start, npts, sensor_type, dt,
                        vel_trcs=None, disp_trcs=None):
    """Peak physical-unit amplitudes in a window of response-corrected traces.

    For accelerometer sensors, `vel_trcs` and `disp_trcs` should be
    pre-computed via `remove_response(output='VEL')` and `output='DISP')`
    -- ObsPy's spectral inversion with water_level handles the integration
    correctly. Using `numpy.cumsum(acc)` here produced amp_vel_nm_s values
    ~60-120x too large because DC drift dominated the integral.
    """
    nan = float('nan')
    if not phys_trcs:
        return nan, nan, nan
    rel_end = min(rel_start + npts, phys_trcs[0].stats.npts)
    if rel_end <= rel_start:
        return nan, nan, nan
    win = np.stack([tr.data[rel_start:rel_end] for tr in phys_trcs],
                   axis=0).astype(np.float64)
    if win.size == 0 or np.any(~np.isfinite(win)):
        return nan, nan, nan

    def _win_of(trcs):
        if not trcs:
            return None
        end = min(rel_start + npts, trcs[0].stats.npts)
        if end <= rel_start:
            return None
        w = np.stack([tr.data[rel_start:end] for tr in trcs],
                     axis=0).astype(np.float64)
        return None if (w.size == 0 or np.any(~np.isfinite(w))) else w

    if sensor_type == 'velocity':
        vel  = win
        disp_w = _win_of(disp_trcs)
        disp = disp_w if disp_w is not None else np.cumsum(
            vel - vel[:, :1], axis=1) * dt
        acc  = np.gradient(vel, dt, axis=1)
    elif sensor_type == 'acceleration':
        acc  = win
        # Prefer spectral-integrated velocity/displacement (ObsPy
        # remove_response output='VEL'/'DISP') over cumsum integration
        # of ACC, which accumulates DC drift.
        vel_w  = _win_of(vel_trcs)
        disp_w = _win_of(disp_trcs)
        vel  = vel_w  if vel_w  is not None else (
            np.cumsum(acc - acc[:, :1], axis=1) * dt)
        disp = disp_w if disp_w is not None else (
            np.cumsum(vel - vel[:, :1], axis=1) * dt)
    else:
        return nan, nan, nan
    return (float(np.max(np.abs(disp))) * 1e9,
            float(np.max(np.abs(vel)))  * 1e9,
            float(np.max(np.abs(acc)))  * 100.0)


def full_response_amplitudes(
    raw_counts, p_idx, p_win, s_idx, s_win,
    mag_start, mag_end,
    inv, net, sta, loc, chn_pre, t0,
):
    """Single-pass full-response computation for one P/S pair.

    Matches station_magnitude_TW() processing chain:
      1. detrend + bandpass 0.075-35 Hz, corners=1, zerophase
      2. remove_response(output='VEL' or 'ACC', water_level=60)
      3. WA simulate(pre_filt=[0.1,0.5,30,35], water_level=60)
      4. CWA station-level quality gates on physical traces
      5. Weak-signal bandpass (1-25 Hz, corners=4) on WA traces

    Returns (wa_mm, p_amps, s_amps) where amps = (disp_nm, vel_nm_s, acc_gal).
    """
    nan3 = (float('nan'),) * 3
    nan_ret = (float('nan'), nan3, nan3)

    chn_pfx = chn_pre.upper()
    if chn_pfx in VEL_CHN_SET:
        output_kind, wa_paz, machine, sensor_type = 'VEL', WA_PAZ_VEL, 'VEL', 'velocity'
    elif chn_pfx in ACC_CHN_SET:
        output_kind, wa_paz, machine, sensor_type = 'ACC', WA_PAZ_ACC, 'ACC', 'acceleration'
    else:
        return nan_ret

    total_npts = raw_counts.shape[1]
    margin = int(WA_MARGIN_SEC * FS)
    win_lo = min(p_idx, s_idx, mag_start)
    win_hi = max(p_idx + p_win, s_idx + s_win, mag_end)
    win_start = max(0, win_lo - margin)
    win_end = min(total_npts, win_hi + margin)
    if win_end - win_start < 100:
        return nan_ret

    dt = 1.0 / FS
    wa_stream = Stream()
    for ci, comp in enumerate(('E', 'N', 'Z')):
        tr = Trace(data=raw_counts[ci, win_start:win_end].copy())
        tr.stats.network       = net
        tr.stats.station       = sta
        tr.stats.location      = loc
        tr.stats.channel       = f'{chn_pre}{comp}'
        tr.stats.sampling_rate = FS
        tr.stats.starttime     = t0 + win_start / FS
        wa_stream.append(tr)

    try:
        wa_stream = wa_stream.detrend('simple').filter(
            'bandpass', freqmin=0.075, freqmax=35.0, corners=1, zerophase=True)
    except Exception:
        return nan_ret

    phys_trcs = Stream()
    wa_trcs   = Stream()
    # For accel sensors we will also request spectral-integrated VEL/DISP.
    # cumsum integration of ACC accumulates DC drift -> amp_vel_nm_s was
    # 60-120x too large vs velocity sensors. Spectral inversion (ObsPy
    # remove_response with water_level=60) handles the 1/(jw) factor
    # correctly.
    vel_trcs  = Stream() if sensor_type == 'acceleration' else None
    disp_trcs = Stream() if sensor_type == 'acceleration' else None
    # Track the (channel, location) that worked for each component so we
    # don't redo the fallback search for VEL/DISP calls.
    resolved: list[tuple[int, str, str]] = []
    try:
        for ci, tr in enumerate(wa_stream):
            chn0 = tr.stats.channel
            loc0 = tr.stats.location
            ok = False
            chn_used, loc_used = chn0, loc0
            chns = CHN_FALLBACKS.get(chn0, [chn0])
            locs = [loc0] + [lc for lc in ('', '--') if lc != loc0]
            for c in chns:
                for lc in locs:
                    tr.stats.channel  = c
                    tr.stats.location = lc
                    try:
                        tr.remove_response(inventory=inv, output=output_kind,
                                           water_level=60)
                        ok = True
                        chn_used, loc_used = c, lc
                        break
                    except Exception:
                        continue
                if ok:
                    break
            if not ok:
                tr.stats.channel  = chn0
                tr.stats.location = loc0
                return nan_ret
            phys_trcs.append(tr.copy())
            resolved.append((ci, chn_used, loc_used))
            try:
                tr.simulate(paz_remove=None, paz_simulate=wa_paz,
                            water_level=60, pre_filt=[0.1, 0.5, 30.0, 35.0],
                            taper=True, taper_fraction=0.005)
            except Exception:
                return nan_ret
            wa_trcs.append(tr)
    except Exception:
        return nan_ret

    if len(wa_trcs) == 0 or len(phys_trcs) == 0:
        return nan_ret

    # Accelerometer spectral integration - second pass to get VEL & DISP
    # with correct low-frequency handling (water_level=60 prevents DC blow-up).
    if sensor_type == 'acceleration':
        for ci, chn_used, loc_used in resolved:
            try:
                for out, out_stream in (('VEL', vel_trcs),
                                        ('DISP', disp_trcs)):
                    tr2 = Trace(data=raw_counts[ci, win_start:win_end].copy())
                    tr2.stats.network       = net
                    tr2.stats.station       = sta
                    tr2.stats.location      = loc_used
                    tr2.stats.channel       = chn_used
                    tr2.stats.sampling_rate = FS
                    tr2.stats.starttime     = t0 + win_start / FS
                    tr2.detrend('simple').filter(
                        'bandpass', freqmin=0.075, freqmax=35.0,
                        corners=1, zerophase=True)
                    tr2.remove_response(inventory=inv, output=out,
                                        water_level=60)
                    out_stream.append(tr2)
            except Exception:
                # On any failure, fall back to cumsum (still better than nothing).
                vel_trcs  = None
                disp_trcs = None
                break

    # CWA station-level quality gates
    # NOTE: for continuous picking, the ACC noise-floor gate (0.1 gal) is
    # skipped - many legitimate small events fall below it.  The VEL
    # clipping gate and weak-signal bandpass are kept because they fix
    # data problems (saturation, microseism contamination), not signal
    # strength.  Downstream magnitude code can apply stricter gates once
    # the event location and distance are known.
    if machine == 'VEL':
        max_vel = max(float(np.max(np.abs(tr.data))) for tr in phys_trcs)
        if chn_pfx == 'EH' and max_vel > 0.01:
            return nan_ret
        weak_thr = 3e-6 if chn_pfx == 'EH' else 5e-6
        if max_vel < weak_thr:
            try:
                wa_trcs.filter('bandpass', freqmin=1.0, freqmax=25.0,
                               corners=4, zerophase=True)
            except Exception:
                pass

    # WA: horizontal mean of per-channel max in magnitude window
    rel_lo = mag_start - win_start
    rel_hi = min(mag_end - win_start, wa_trcs[0].stats.npts)
    if rel_hi <= rel_lo:
        wa_mm = float('nan')
    else:
        horiz = [tr for tr in wa_trcs if tr.stats.channel[-1] in '12NE']
        if not horiz:
            wa_mm = float('nan')
        else:
            wa_mm = float(np.mean(
                [float(np.max(np.abs(tr.data[rel_lo:rel_hi]))) for tr in horiz]
            )) * 1000.0

    p_amps = phys_amps_in_window(phys_trcs, p_idx - win_start, p_win, sensor_type, dt,
                                 vel_trcs=vel_trcs, disp_trcs=disp_trcs)
    s_amps = phys_amps_in_window(phys_trcs, s_idx - win_start, s_win, sensor_type, dt,
                                 vel_trcs=vel_trcs, disp_trcs=disp_trcs)
    return wa_mm, p_amps, s_amps
