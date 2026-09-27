"""INT8 static PTQ (QDQ, per-channel) of the EdgeRP90 ONNX + fp32-vs-INT8 latency and
fidelity. Latency measured on x86 CPU @ 4 threads (indicative Pi-4 proxy, NOT a Pi
number). Calibration/test mix random-noise and synthetic-pulse inputs so activation
ranges cover both quiet and event-like responses.

The fidelity check here (INT8 against fp32 argmax agreement) is a quick sanity check.
Measuring INT8 accuracy needs the INT8 model scored on labelled waveforms.
"""
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod,
)

D = Path("edge_model_design/deploy")
FP32 = D / "edge_rp90_e189.onnx"
INT8 = D / "edge_rp90_e189_int8.onnx"
rng = np.random.default_rng(0)


def make_input(event=False):
    x = rng.standard_normal((1, 3, 9000)).astype(np.float32)
    if event:  # add a damped-oscillation "arrival" to elicit picker/detector response
        t0 = rng.integers(1000, 8000)
        t = np.arange(0, 600)
        pulse = (np.sin(2 * np.pi * 0.05 * t) * np.exp(-t / 150)).astype(np.float32)
        x[0, :, t0:t0 + 600] += 6.0 * pulse
    z = (rng.random((1, 1, 9000), dtype=np.float32) * 2 - 1)
    return {"x": x, "z_raw": z}


class CalibReader(CalibrationDataReader):
    def __init__(self, n=64):
        self.it = iter([make_input(event=(i % 2 == 0)) for i in range(n)])

    def get_next(self):
        return next(self.it, None)


print("Quantizing (static PTQ, QDQ, per-channel weights, INT8 act/weight)...")
quantize_static(
    str(FP32), str(INT8), CalibReader(64),
    quant_format=QuantFormat.QDQ, per_channel=True,
    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.MinMax,
)
print(f"  fp32 {FP32.stat().st_size/1e6:.2f} MB  ->  int8 {INT8.stat().st_size/1e6:.2f} MB")


def sess(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4  # Pi-4 has 4 cores
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def bench(path, n=30):
    s = sess(path)
    inp = make_input(event=True)
    for _ in range(5):
        s.run(None, inp)
    t = time.perf_counter()
    for _ in range(n):
        s.run(None, inp)
    return (time.perf_counter() - t) / n * 1000  # ms/window


lat_fp32 = bench(FP32)
lat_int8 = bench(INT8)
print(f"\nx86 CPU latency (4 threads, ms per 90 s window):")
print(f"  fp32  {lat_fp32:7.1f} ms")
print(f"  int8  {lat_int8:7.1f} ms   ({lat_fp32/lat_int8:.2f}x vs fp32)")

# fidelity: INT8 vs fp32 on 24 event traces
sf, si = sess(FP32), sess(INT8)
pk_agree = det_agree = 0.0
maxdiff = {0: 0.0, 1: 0.0, 2: 0.0}
N = 24
for _ in range(N):
    inp = make_input(event=True)
    of = sf.run(None, inp)
    oi = si.run(None, inp)
    for h in range(3):
        maxdiff[h] = max(maxdiff[h], float(np.abs(of[h] - oi[h]).max()))
    pk_agree += float((of[0].argmax(1) == oi[0].argmax(1)).mean())
    det_agree += float((of[2].argmax(1) == oi[2].argmax(1)).mean())
print(f"\nINT8 vs fp32 fidelity over {N} event traces:")
print(f"  picker argmax agreement:   {pk_agree/N*100:.2f}%")
print(f"  detector argmax agreement: {det_agree/N*100:.2f}%")
print(f"  max|prob diff|  picker {maxdiff[0]:.3f}  polarity {maxdiff[1]:.3f}  detector {maxdiff[2]:.3f}")
