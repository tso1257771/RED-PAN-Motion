"""INT8 static PTQ (QDQ, per-channel) of the on-device edge_rp90 export, so the
INT8 graph and the fp32 graph benchmarked here come from the same checkpoint."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantType, QuantFormat, CalibrationMethod)
import os
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                                   # scripts/benchmarks/device -> repo root
ONNX_DIR = Path(os.environ.get("RPM_ONNX_DIR", HERE / "onnx"))
RESULTS_DIR = Path(os.environ.get("RPM_RESULTS_DIR", HERE / "results"))
CKPT_DIR = Path(os.environ.get("RPM_CKPT_DIR", REPO / "checkpoints"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


D = ONNX_DIR
FP32, INT8 = D / "edge_rp90.onnx", D / "edge_rp90_int8.onnx"
rng = np.random.default_rng(0)


def make_input(event=False):
    x = rng.standard_normal((1, 3, 9000)).astype(np.float32)
    if event:
        t0 = int(rng.integers(1000, 8000)); t = np.arange(600)
        x[0, :, t0:t0 + 600] += 6.0 * (np.sin(2 * np.pi * 0.05 * t) * np.exp(-t / 150)).astype(np.float32)
    return {"x": x, "z_raw": (rng.random((1, 1, 9000), dtype=np.float32) * 2 - 1)}


class Calib(CalibrationDataReader):
    def __init__(self, n=64):
        self.it = iter([make_input(event=(i % 2 == 0)) for i in range(n)])
    def get_next(self):
        return next(self.it, None)


quantize_static(str(FP32), str(INT8), Calib(64), quant_format=QuantFormat.QDQ,
                per_channel=True, weight_type=QuantType.QInt8,
                activation_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)
print(f"fp32 {FP32.stat().st_size/1e6:.2f} MB -> int8 {INT8.stat().st_size/1e6:.2f} MB "
      f"({FP32.stat().st_size/INT8.stat().st_size:.2f}x)")
