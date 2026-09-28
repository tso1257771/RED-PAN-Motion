"""Amplitude and sensitivity helpers on velocity sensors and multi-epoch stations."""
import numpy as np
import obspy
from obspy import Stream, Trace, UTCDateTime
from obspy.core.inventory import Channel, Inventory, Network, Response, Station
from obspy.core.inventory.response import InstrumentSensitivity

from redpan_motion.amplitudes import full_response_amplitudes
from redpan_motion.waveform_io import apply_sensitivity


def test_full_response_amplitudes_runs_on_a_velocity_sensor():
    # 0.1.0 raised NameError ('chn_pfx' is not defined) at the velocity-sensor
    # gate, so no HH, BH or EH record ever got amplitudes.
    inv = obspy.read_inventory()                   # GR.FUR HH*: full responses
    raw = np.random.default_rng(0).normal(0, 200, (3, 6000))
    raw[:, 2000:2600] *= 50
    wa_mm, p_amps, s_amps = full_response_amplitudes(
        raw, 2000, 500, 3000, 1000, 1500, 5000,
        inv, "GR", "FUR", "", "HH", UTCDateTime(2020, 1, 1))
    assert np.isfinite(wa_mm) and wa_mm > 0
    assert all(np.isfinite(v) and v > 0 for v in p_amps + s_amps)


def _two_epoch_inventory():
    def channel(start, end, sensitivity):
        return Channel(
            code="HHZ", location_code="", latitude=45.0, longitude=26.0,
            elevation=0.0, depth=0.0, start_date=UTCDateTime(start),
            end_date=UTCDateTime(end) if end else None,
            response=Response(instrument_sensitivity=InstrumentSensitivity(
                value=sensitivity, frequency=1.0, input_units="M/S",
                output_units="COUNTS")))
    station = Station(code="ISR", latitude=45.0, longitude=26.0, elevation=0.0,
                      channels=[channel("2008-01-01", "2009-01-01", 4.0e8),
                                channel("2009-01-01", None, 8.0e8)])
    return Inventory(networks=[Network(code="RO", stations=[station])])


def _trace(starttime):
    tr = Trace(np.full(100, 8.0e8))
    tr.stats.network, tr.stats.station, tr.stats.channel = "RO", "ISR", "HHZ"
    tr.stats.starttime = UTCDateTime(starttime)
    return tr


def test_sensitivity_comes_from_the_epoch_that_recorded_the_trace():
    # 0.1.0 took the first epoch with a response, here the 2008 sensor, and
    # was a factor 2 off for every trace recorded since 2009.
    st = Stream([_trace("2023-03-12T17:43:00")])
    assert apply_sensitivity(st, _two_epoch_inventory())
    np.testing.assert_allclose(st[0].data, 1.0)


def test_a_trace_outside_every_epoch_keeps_the_old_first_match():
    st = Stream([_trace("2005-06-01T00:00:00")])
    assert apply_sensitivity(st, _two_epoch_inventory())
    np.testing.assert_allclose(st[0].data, 2.0)          # 8e8 / 4e8
