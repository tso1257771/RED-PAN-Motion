"""load() must align channels whose traces start at different times (day files from
FDSN / S3 begin each channel at its own record boundary)."""
import numpy as np
import obspy
from obspy import Trace, Stream, UTCDateTime

from redpan_motion import load, CH_Z


def test_load_aligns_staggered_channel_starts(tmp_path):
    sr, t0 = 100.0, UTCDateTime("2023-01-03T00:00:00")
    spike_time = t0 + 30.0                      # a spike on Z at a known absolute time
    traces = []
    for comp, start_offset in (("E", 0.0), ("N", -0.71), ("Z", 2.29)):
        n = int(60 * sr)
        data = np.zeros(n, dtype=np.int32)
        start = t0 + start_offset
        if comp == "Z":
            data[int(round((spike_time - start) * sr))] = 1000
        tr = Trace(data, header=dict(network="NZ", station="TEST", location="10", channel="EH" + comp,
                                     sampling_rate=sr, starttime=start))
        traces.append(tr)
    f = tmp_path / "NZ.TEST.10.EH?.mseed"
    Stream(traces).write(str(f), format="MSEED", encoding="STEIM2")

    raw, _, _, ids, t_start, st = load(str(f))
    i_spike = int(np.argmax(np.abs(raw[CH_Z])))
    recovered = t_start + i_spike / sr
    assert abs(recovered - spike_time) < 0.5 / sr, f"Z spike recovered at {recovered}, expected {spike_time}"
    assert raw.shape[1] >= int(60 * sr)        # nothing lost: padded to the union of the channels


def test_load_ignores_zero_length_traces(tmp_path):
    sr, t0 = 100.0, UTCDateTime("2023-08-29T00:00:00")
    traces = [Trace(np.ones(int(60 * sr), dtype=np.int32) * k, header=dict(network="NZ", station="HNCZ", location="10",
                    channel=ch, sampling_rate=sr, starttime=t0)) for k, ch in ((1, "HH1"), (2, "HH2"), (3, "HHZ"))]
    traces.append(Trace(np.zeros(0, dtype=np.int32), header=dict(network="NZ", station="HNCZ", location="10",
                        channel="HHN", sampling_rate=sr, starttime=t0)))   # empty renamed channel
    f = tmp_path / "NZ.HNCZ.10.HH?.mseed"
    Stream(traces).write(str(f), format="MSEED", encoding="STEIM2")
    raw, _, _, _, _, _ = load(str(f))
    assert raw.shape == (3, int(60 * sr))
    assert raw[CH_Z].max() == 3


def test_load_prefers_the_channel_with_data_when_codes_collide(tmp_path):
    """A one-sample stub under the new channel name (HHN) must not overwrite the real
    horizontal (HH2) that maps to the same component row."""
    sr, t0 = 100.0, UTCDateTime("2023-08-29T00:00:00")
    hdr = dict(network="NZ", station="HNCZ", location="10", sampling_rate=sr, starttime=t0)
    traces = [Trace(np.ones(int(60 * sr), dtype=np.int32) * k, header=dict(hdr, channel=ch))
              for k, ch in ((1, "HH1"), (2, "HH2"), (3, "HHZ"))]
    traces.append(Trace(np.array([7], dtype=np.int32), header=dict(hdr, channel="HHN")))   # 1-sample stub
    f = tmp_path / "NZ.HNCZ.10.HH?.mseed"
    Stream(traces).write(str(f), format="MSEED", encoding="STEIM2")
    raw, _, _, _, _, _ = load(str(f))
    assert raw.shape == (3, int(60 * sr))
    assert raw[1].min() == 2 and raw[1].max() == 2      # N row still carries HH2, not the stub
