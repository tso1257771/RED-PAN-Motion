"""picks_to_dataframe writes full-precision values and the opt-in WA_mm_E/N/Z columns (0.1.8)."""
import io

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime

from redpan_motion.picks import COLUMNS, WA_COMPONENT_COLUMNS, picks_to_dataframe

T, DT = 6000, 0.01
P_IDX, S_IDX = 2000, 3000
P_PEAK, S_PEAK, UP = np.float32(0.8765432), np.float32(0.6543219), np.float32(0.7123456)


def _model_output(polarity=False):
    x = np.arange(T)
    picker = np.zeros((T, 3), np.float32)
    picker[:, 0] = P_PEAK * np.exp(-0.5 * ((x - P_IDX) / 20.0) ** 2)
    picker[:, 1] = S_PEAK * np.exp(-0.5 * ((x - S_IDX) / 20.0) ** 2)
    picker[:, 2] = 1 - picker[:, 0] - picker[:, 1]
    detector = np.zeros((T, 2), np.float32)
    detector[1800:3600, 0] = 0.95
    detector[:, 1] = 1 - detector[:, 0]
    pol = None
    if polarity:
        pol = np.zeros((T, 3), np.float32)
        pol[:, 0], pol[:, 1] = 0.1, UP
        pol[:, 2] = 1 - pol[:, 0] - pol[:, 1]
    return picker, detector, pol


def test_values_are_written_at_full_precision():
    # 0.1.7 rounded prob to 2 and polarity_prob to 3 decimals and cut timestamps to 0.1 ms.
    picker, detector, pol = _model_output(polarity=True)
    t0 = UTCDateTime("2023-09-07T00:00:00.008393")
    df = picks_to_dataframe(picker, detector, pol, t0, "NZ.KUZ.10.HH", dt=DT, amplitude=False)
    assert list(df.columns) == COLUMNS
    p, s = df[df.type == "p"].iloc[0], df[df.type == "s"].iloc[0]
    assert p.prob == float(P_PEAK) and s.prob == float(S_PEAK)
    assert p.timestamp == "2023-09-07T00:00:20.008393"
    assert UTCDateTime(s.timestamp) == t0 + S_IDX * DT
    assert p.polarity == 1 and p.polarity_prob == float(UP)
    back = pd.read_csv(io.StringIO(df.to_csv(index=False)))
    assert back.prob.tolist() == df.prob.tolist() and back.timestamp.tolist() == df.timestamp.tolist()


def test_wa_components_come_from_the_same_magnitude_window():
    inv = obspy.read_inventory()                   # GR.FUR HH*: full responses
    raw = np.random.default_rng(0).normal(0, 200, (3, T))
    raw[:, P_IDX:P_IDX + 1500] *= 50
    picker, detector, _ = _model_output()
    t0 = UTCDateTime(2020, 1, 1)
    kw = dict(raw_counts=raw, wf_sensonly=raw.astype(np.float64), inv=inv, dt=DT)
    df = picks_to_dataframe(picker, detector, None, t0, "GR.FUR..HH", wa_components=True, **kw)
    assert list(df.columns) == COLUMNS + WA_COMPONENT_COLUMNS
    row = df.iloc[0]
    assert all(np.isfinite(row[c]) and row[c] > 0 for c in WA_COMPONENT_COLUMNS)
    assert np.isclose(row.WA_amp_disp_mm, (row.WA_mm_E + row.WA_mm_N) / 2)   # horizontal mean
    plain = picks_to_dataframe(picker, detector, None, t0, "GR.FUR..HH", **kw)
    assert list(plain.columns) == COLUMNS
    pd.testing.assert_frame_equal(plain, df[COLUMNS])


def test_wa_components_are_nan_without_amplitudes():
    picker, detector, _ = _model_output()
    df = picks_to_dataframe(picker, detector, None, UTCDateTime(2020, 1, 1), "GR.FUR..HH", dt=DT,
                            amplitude=False, wa_components=True)
    assert list(df.columns) == COLUMNS + WA_COMPONENT_COLUMNS
    assert len(df) == 2 and df[WA_COMPONENT_COLUMNS].isna().all().all()
