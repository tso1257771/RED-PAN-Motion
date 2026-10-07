"""Compare the torch RP60 port against the PRODUCTION tf2onnx graph, running the
production graph through the test Triton server (a local ORT session on it stalls
in graph optimization; Triton loads it in seconds)."""
import http.client, os, sys
from pathlib import Path
import numpy as np, onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parent))
from triton_bench import build_request, infer, parse_outputs, SPECS
import os
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                                   # scripts/benchmarks/device -> repo root
ONNX_DIR = Path(os.environ.get("RPM_ONNX_DIR", HERE / "onnx"))
RESULTS_DIR = Path(os.environ.get("RPM_RESULTS_DIR", HERE / "results"))
CKPT_DIR = Path(os.environ.get("RPM_CKPT_DIR", REPO / "checkpoints"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


so = ort.SessionOptions(); so.log_severity_level = 3; so.intra_op_num_threads = 1
mine = ort.InferenceSession(str(ONNX_DIR / "redpan_60s.onnx"), so, providers=["CPUExecutionProvider"])
conn = http.client.HTTPConnection(os.environ.get("RPM_TRITON_HOST", "localhost"),
                                  int(os.environ.get("RPM_TRITON_PORT", "7183")), timeout=300)
spec = SPECS["onnx_RP60_ref"]
rng = np.random.default_rng(7); worst = {}
for i in range(3):
    ntc = rng.standard_normal((1, 6000, 3)).astype(np.float32)
    h, p = build_request("onnx_RP60_ref", {"input": np.ascontiguousarray(ntc)}, spec)
    prod = parse_outputs(*infer(conn, "onnx_RP60_ref", h, p))
    a = mine.run(None, {"x": np.ascontiguousarray(ntc.transpose(0, 2, 1))})
    for n, u in zip(("picker", "detector"), a):
        worst[n] = max(worst.get(n, 0.0), float(np.abs(u.transpose(0, 2, 1) - prod[n]).max()))
    print(f"  trial {i} done", flush=True)
print("torch port vs PRODUCTION tf2onnx graph (identical inputs):", flush=True)
for n, d in worst.items():
    print(f"  {n:9s} max|diff| = {d:.3e}  -> {'SAME weights' if d < 1e-4 else 'DIFFERENT model/checkpoint'}", flush=True)
