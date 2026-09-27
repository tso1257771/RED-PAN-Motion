"""
PyTorch implementation of RED-PAN (RED-PAN-Motion).

Model loading + inference plus picks / amplitudes / response / waveform I/O as
importable modules (``from redpan_motion.X import ...`` after ``pip install -e .``).

Quick Start:
    from redpan_motion import REDPANPredictor, load, highpass, picks_to_dataframe

    raw, wf_so, inv, (net, sta, loc, chn), t0, st = load("STA.NET.LOC.HH?.YYYY.JJJ", xml=None)
    pred = REDPANPredictor.from_checkpoint("redpan_motion", device="cuda")
    picker, detector, polarity = pred.predict_arrays(highpass(raw, 1.0))
    df = picks_to_dataframe(picker, detector, polarity, t0, f"{net}.{sta}.{loc}.{chn}",
                            amplitude=False)            # -> redpan_picks DataFrame
"""

from redpan_motion.models import (
    MTAN_R2UNet_RP90_Motion,
    build_mtan_r2unet_rp90_motion,
    MTAN_R2UNet,
)
from redpan_motion.inference import REDPANPredictor
from redpan_motion.training import REDPANTrainer, TrainingConfig, MultiTaskLoss
from redpan_motion.waveform_io import (
    load, bandpass_for_model, highpass, apply_sensitivity, group_instruments, CH_Z,
)
from redpan_motion.picks import (
    picks_to_dataframe, detect_events_joint, s_match_window, match_s_arrival, COLUMNS,
)
from redpan_motion.amplitudes import (
    window_amplitudes, full_response_amplitudes, adaptive_s_window, compute_snr,
)
from redpan_motion.response import sensor_type, CHN_FALLBACKS

__all__ = [
    # models
    'MTAN_R2UNet_RP90_Motion', 'build_mtan_r2unet_rp90_motion', 'MTAN_R2UNet',
    # inference
    'REDPANPredictor',
    # training
    'REDPANTrainer', 'TrainingConfig', 'MultiTaskLoss',
    # waveform I/O + preprocessing
    'load', 'bandpass_for_model', 'highpass', 'apply_sensitivity', 'group_instruments', 'CH_Z',
    # picks
    'picks_to_dataframe', 'detect_events_joint', 's_match_window', 'match_s_arrival', 'COLUMNS',
    # amplitudes + response (vendored from RED-PAN)
    'window_amplitudes', 'full_response_amplitudes', 'adaptive_s_window', 'compute_snr',
    'sensor_type', 'CHN_FALLBACKS',
]

__version__ = '0.1.0'
