"""Re-export RED-PAN 60 s with an ONNX-friendly nearest upsample.

redpan_60s.py:133 uses ``x.repeat_interleave(k, dim=2)`` for Keras UpSampling1D.
torch 1.13 traces that into a dynamic Loop over a tensor sequence: the exported
graph got 8 Loop/SequenceInsert regions that cost ~88% of its runtime (9.2 s per
60 s window on this Jetson, vs ~0.1 s expected from its 0.082 GMAC).

Expand+Reshape is elementwise-identical for an integer factor on a 3-D tensor and
exports as two cheap ops. The reference outputs here are computed BEFORE the patch
is installed, so the parity check proves the swap changes no numerics.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
import os
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                                   # scripts/benchmarks/device -> repo root
ONNX_DIR = Path(os.environ.get("RPM_ONNX_DIR", HERE / "onnx"))
RESULTS_DIR = Path(os.environ.get("RPM_RESULTS_DIR", HERE / "results"))
CKPT_DIR = Path(os.environ.get("RPM_CKPT_DIR", REPO / "checkpoints"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


import math as _math
if not hasattr(_math, "prod"):
    import functools as _ft, operator as _op
    _math.prod = lambda it, start=1: _ft.reduce(_op.mul, it, start)
import types as _types
ROOT = str(REPO)
for _n, _p in (("redpan_motion", f"{ROOT}/redpan_motion"),
               ("redpan_motion.models", f"{ROOT}/redpan_motion/models")):
    _m = _types.ModuleType(_n); _m.__path__ = [_p]; sys.modules[_n] = _m
sys.path.insert(0, ROOT)
from redpan_motion.models.redpan_60s import build_redpan_60s

CK = CKPT_DIR / "redpan_60s"
OUT = ONNX_DIR

cfg = json.load(open(CK / "config.json"))
raw = torch.load(CK / "best.pt", map_location="cpu")
sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
model = build_redpan_60s(input_size=tuple(cfg["input_size"]))
model.load_state_dict(sd)
model.eval()

torch.manual_seed(0)
x = torch.randn(1, 3, 6000)
with torch.no_grad():                       # reference from the UNPATCHED model
    ref = [t.numpy() for t in model(x)]

_orig = torch.Tensor.repeat_interleave


def _ri(self, repeats, dim=None, **kw):
    """repeat_interleave(int, dim=2) on (B,C,T) -> Expand+Reshape; else original."""
    if isinstance(repeats, int) and dim in (2, -1) and self.dim() == 3:
        B, C, T = self.shape
        return self.unsqueeze(-1).expand(B, C, T, repeats).reshape(B, C, T * repeats)
    return _orig(self, repeats, dim=dim, **kw) if dim is not None else _orig(self, repeats, **kw)


torch.Tensor.repeat_interleave = _ri
with torch.no_grad():                       # patched model, same weights
    patched = [t.numpy() for t in model(x)]
print("torch patched vs torch original:",
      " ".join(f"{np.abs(a - b).max():.2e}" for a, b in zip(patched, ref)))


class Wrap2(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        p, d = self.m(x)
        return p, d


path = OUT / "redpan_60s.onnx"
torch.onnx.export(Wrap2(model), (x,), str(path), input_names=["x"],
                  output_names=["picker", "detector"], opset_version=17,
                  do_constant_folding=True)
torch.Tensor.repeat_interleave = _orig

import onnx, collections
g = onnx.load(str(path)).graph
c = collections.Counter(n.op_type for n in g.node)
sus = {k: v for k, v in c.items() if "Sequence" in k or k == "Loop"}
sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
got = sess.run(None, {"x": x.numpy()})
print(f"re-exported {path.stat().st_size/1e6:.2f} MB  nodes {len(g.node)}  Conv {c.get('Conv',0)}  "
      f"loop/sequence ops {sus if sus else 'none'}")
print("ORT vs ORIGINAL torch model:",
      " ".join(f"{np.abs(a - b).max():.2e}" for a, b in zip(got, ref)))
