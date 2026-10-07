#!/usr/bin/env python
"""INT8 vs FP32 P/S pick F1 of Edge-RED-PAN-Motion (Section V-D INT8 check).

1. FP32 graph: exported from the shipped ``edge_rp90`` checkpoint (opset 17, inputs x (1,3,9000),
   z_raw (1,1,9000); outputs picker, polarity, detector), unless ``--fp32-onnx`` is given.
2. INT8 graph: ONNX Runtime static post-training quantization, QDQ, per-channel QInt8 weights and
   QInt8 activations, calibrated on 192 held-out records with ``--calib`` (minmax | percentile),
   unless ``--int8-onnx`` gives an existing graph (e.g. the one timed on the device).
3. Records: STEAD 90 s HDF5 test split; calibration / evaluation lists in ``data/int8/`` (drawn with a
   seeded shuffle, seed 0: 192 calibration, then 2,000 evaluation records whose first 400 are the
   n = 400 set). Input: demean, 5% taper, 1 Hz zero-phase high-pass, per-channel z-score; z_raw = raw
   vertical / max|raw vertical|.
4. P and S pick F1 at probability 0.3, tolerance 0.5 s (P) and 1.0 s (S), scored two ways:
   argmax (one pick per record) and peaks (every local maximum >= 0.3, >= 1 s apart; extra picks
   are false positives).

Writes ``<results_root>/int8/<tag>/int8_pick_f1.json`` and per_record.csv.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rpm_bench import config
from rpm_bench.constants import COMMON_THR, DT, P_TOL_SEC, S_TOL_SEC
from rpm_bench.results import write_json
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.signal.windows import tukey

HERE = Path(__file__).resolve().parent
SOS = butter(4, 1.0, btype="highpass", fs=100.0, output="sos")
TAPER = tukey(9000, 0.05).astype(np.float32)
THR = COMMON_THR
TOL = {"P": round(P_TOL_SEC / DT), "S": round(S_TOL_SEC / DT)}  # samples: 50 and 100


def first_sample(cell) -> int:
    """First arrival sample of a list cell such as ``"[752]"``. (Not rpm_bench.datasets._parse_arrival,
    which differs on non-string and malformed cells.)"""
    return int(str(cell).strip("[]").split(",")[0])


def export_fp32(path: Path):
    """Export the shipped edge_rp90 checkpoint to an FP32 ONNX graph (opset 17)."""
    import torch

    from redpan_motion.inference import REDPANPredictor

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, z_raw):
            picker, polarity, detector = self.m(x, z_raw=z_raw)
            return picker, polarity, detector

    w = Wrap(REDPANPredictor.from_checkpoint("edge_rp90", device="cpu").model.eval()).eval()
    x, z = torch.randn(1, 3, 9000), torch.randn(1, 1, 9000)
    with (
        warnings.catch_warnings()
    ):  # the tracer warns about Python control flow; the graph is checked by its F1
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        torch.onnx.export(
            w,
            (x, z),
            str(path),
            input_names=["x", "z_raw"],
            output_names=["picker", "polarity", "detector"],
            opset_version=17,
            do_constant_folding=True,
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument(
        "--calib", default="minmax", choices=["minmax", "percentile"], help="calibration method"
    )
    p.add_argument(
        "--fp32-onnx", type=Path, default=None, help="existing FP32 graph (default: export one)"
    )
    p.add_argument(
        "--int8-onnx", type=Path, default=None, help="score this INT8 graph instead of quantizing"
    )
    p.add_argument(
        "--records-dir",
        type=Path,
        default=HERE / "data" / "int8",
        help="calibration_192.csv / evaluation_2000.csv (sample_id, hdf5_index, ...)",
    )
    p.add_argument(
        "--tag", default=None, help="output sub-directory (default: the calibration method)"
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    cfg = config.load(args.config)
    out = config.results_dir(
        cfg, "int8", args.tag or ("given_int8" if args.int8_onnx else args.calib)
    )
    fp32 = args.fp32_onnx or out / "edge_rp90_fp32.onnx"
    if not fp32.exists():
        export_fp32(fp32)
    calib = pd.read_csv(args.records_dir / "calibration_192.csv")
    ev = pd.read_csv(args.records_dir / "evaluation_2000.csv")
    int8 = args.int8_onnx or out / f"edge_rp90_int8_{args.calib}.onnx"
    with h5py.File(cfg["h5_root"] / "STEAD" / "STEAD_dataset_90s_singleEQ.h5", "r") as h5:
        WF = h5["/singleEQ/test/waveforms"]

        def prep(idx):
            raw = WF[int(idx)].astype(np.float32)
            x = raw - raw.mean(axis=1, keepdims=True)
            x = sosfiltfilt(SOS, x * TAPER, axis=1).astype(np.float32)
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-10)
            z = raw[2:3]
            zm = float(np.abs(z).max())
            return {
                "x": x[None].astype(np.float32),
                "z_raw": (z / zm if zm > 1e-9 else z)[None].astype(np.float32),
            }

        class Calib(CalibrationDataReader):
            def __init__(self):
                self.it = iter([prep(i) for i in calib.hdf5_index])

            def get_next(self):
                return next(self.it, None)

        if not args.int8_onnx:
            quantize_static(
                str(fp32),
                str(int8),
                Calib(),
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QInt8,
                calibrate_method={
                    "minmax": CalibrationMethod.MinMax,
                    "percentile": CalibrationMethod.Percentile,
                }[args.calib],
            )
        so = ort.SessionOptions()
        so.log_severity_level = 3
        S = {
            k: ort.InferenceSession(str(g), so, providers=["CPUExecutionProvider"])
            for k, g in (("fp32", fp32), ("int8", int8))
        }
        rows = []
        for _, r in ev.iterrows():
            inp = prep(r.hdf5_index)
            L = {"P": first_sample(r.p_arrival_sample), "S": first_sample(r.s_arrival_sample)}
            outs = {k: s.run(["picker"], inp)[0][0] for k, s in S.items()}
            rec = {
                "sample_id": r.sample_id,
                "max_abs_diff": float(np.abs(outs["fp32"][:2] - outs["int8"][:2]).max()),
            }
            for k, pk in outs.items():
                for ph, ch in (("P", 0), ("S", 1)):
                    prob = pk[ch]
                    peaks, _ = find_peaks(prob, height=THR, distance=100)
                    hit = np.abs(peaks - L[ph]) <= TOL[ph]
                    rec[f"{k}_{ph}_peaks_tp"] = int(hit.any())
                    rec[f"{k}_{ph}_peaks_fp"] = int(len(peaks) - int(hit.any()))
                    a = int(prob.argmax())
                    ok = prob[a] >= THR
                    rec[f"{k}_{ph}_argmax_tp"] = int(ok and abs(a - L[ph]) <= TOL[ph])
                    rec[f"{k}_{ph}_argmax_fp"] = int(ok and abs(a - L[ph]) > TOL[ph])
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_record.csv", index=False)

    def f1(sub, k, ph, mode):
        tp = sub[f"{k}_{ph}_{mode}_tp"].sum()
        fp = sub[f"{k}_{ph}_{mode}_fp"].sum()
        return float(2 * tp / (2 * tp + fp + len(sub) - tp))

    res = {
        "fp32_MB": fp32.stat().st_size / 1e6,
        "int8_MB": Path(int8).stat().st_size / 1e6,
        "onnxruntime": ort.__version__,
        "calib": None if args.int8_onnx else args.calib,
        "int8_graph": str(int8),
    }
    for n in (400, 2000):
        sub = df.iloc[:n]
        for mode in ("argmax", "peaks"):
            v = {f"{k}_{ph}": f1(sub, k, ph, mode) for k in ("fp32", "int8") for ph in ("P", "S")}
            v.update(
                {f"delta_{ph}_pp": 100 * (v[f"int8_{ph}"] - v[f"fp32_{ph}"]) for ph in ("P", "S")}
            )
            res[f"n{n}_{mode}"] = v
            logging.info(
                "n=%d %-6s FP32 P/S %.4f/%.4f  INT8 %.4f/%.4f  delta %+.2f/%+.2f pp",
                n,
                mode,
                v["fp32_P"],
                v["fp32_S"],
                v["int8_P"],
                v["int8_S"],
                v["delta_P_pp"],
                v["delta_S_pp"],
            )
    write_json(out / "int8_pick_f1.json", res)


if __name__ == "__main__":
    main()
