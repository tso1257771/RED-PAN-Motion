"""Export the three shipped checkpoints to ONNX on-device, for a like-for-like
latency comparison. Builds each model from its sibling config.json and loads the
shipped state_dict directly, so no obspy/scipy is needed (the predictor imports them).

  redpan_60s      (1,3,6000)             -> picker, detector
  redpan_motion   (1,3,9000)+(1,1,9000)  -> picker, polarity, detector
  edge_rp90       (1,3,9000)+(1,1,9000)  -> picker, polarity, detector
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


# This Jetson's newest interpreter is 3.7.1; the package uses math.prod (3.8+).
# Shim it before importing redpan_motion. Export/inference semantics are unchanged.
import math as _math
if not hasattr(_math, "prod"):
    import functools as _ft, operator as _op
    _math.prod = lambda it, start=1: _ft.reduce(_op.mul, it, start)

# Import the model modules WITHOUT executing redpan_motion/__init__.py, which
# imports the predictor (obspy) and the trainer (torch.amp.GradScaler, torch>=2.3).
# This board caps at Python 3.7 / torch 1.13, and none of that is needed to build
# a model and export it. Pre-registering namespace stubs makes the submodules'
# absolute imports resolve while their parents' __init__ bodies never run.
import types as _types
ROOT = str(REPO)
for _name, _path in (("redpan_motion", f"{ROOT}/redpan_motion"),
                     ("redpan_motion.models", f"{ROOT}/redpan_motion/models")):
    _m = _types.ModuleType(_name); _m.__path__ = [_path]; sys.modules[_name] = _m
sys.path.insert(0, ROOT)

from redpan_motion.models.redpan_60s import build_redpan_60s
from redpan_motion.models.mtan_r2unet_rp90_motion import build_mtan_r2unet_rp90_motion
from redpan_motion.models.edge_rp90 import build_edge_rp90

CK = CKPT_DIR
OUT = ONNX_DIR
OUT.mkdir(parents=True, exist_ok=True)
ARCH_KEYS = ('input_size', 'nb_filters', 'strides', 'kernel_size', 'dropout_rate',
             'rrconv_iters', 'polarity_output_channels', 'pol_stream_width_mult',
             'pol_init_rrconv_iters', 'pol_head_ps_att_width', 'pad_mode',
             'block', 'polarity_head')


def load_sd(p):
    raw = torch.load(p, map_location="cpu")
    sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


class Wrap3(torch.nn.Module):
    """(x, z_raw) -> (picker, polarity, detector), the deploy signature."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x, z_raw):
        p, pol, d = self.m(x, z_raw=z_raw)
        return p, pol, d


class Wrap2(torch.nn.Module):
    """(x) -> (picker, detector): RED-PAN 60 s has no polarity head."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        p, d = self.m(x)
        return p, d


def export(name, model, args, in_names, out_names):
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() for b in model.buffers())
    with torch.no_grad():
        ref = model(*args)
    path = OUT / f"{name}.onnx"
    torch.onnx.export(model, args, str(path), input_names=in_names,
                      output_names=out_names, opset_version=17, do_constant_folding=True)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {n: a.numpy() for n, a in zip(in_names, args)})
    diffs = [float(np.abs(g - r.detach().numpy()).max()) for g, r in zip(got, ref)]
    ok = max(diffs) < 1e-4
    print(f"{name:14s} {path.stat().st_size/1e6:5.2f} MB  params {n_par:,} (+{n_buf:,} buf)  "
          f"parity {max(diffs):.2e} {'OK' if ok else 'MISMATCH'}")
    return {"name": name, "onnx": str(path), "mb": path.stat().st_size / 1e6,
            "params": n_par, "buffers": n_buf, "parity_max_abs": max(diffs),
            "inputs": in_names, "outputs": out_names}


meta = []
torch.manual_seed(0)

# --- RED-PAN 60 s ---------------------------------------------------------
cfg = json.load(open(CK / "redpan_60s/config.json"))
m = build_redpan_60s(input_size=tuple(cfg["input_size"]))
m.load_state_dict(load_sd(CK / "redpan_60s/best.pt"))
meta.append(export("redpan_60s", Wrap2(m), (torch.randn(1, 3, 6000),),
                   ["x"], ["picker", "detector"]))

# --- RED-PAN-Motion 90 s --------------------------------------------------
cfg = json.load(open(CK / "redpan_motion/config.json"))
sd = load_sd(CK / "redpan_motion/best.pt")
kw = {k: cfg[k] for k in ARCH_KEYS if k in cfg}
kw["use_polarity"] = any(k.startswith("pol_init_rrconv.") for k in sd)
m = build_mtan_r2unet_rp90_motion(**kw)
m.load_state_dict(sd)
meta.append(export("redpan_motion", Wrap3(m), (torch.randn(1, 3, 9000), torch.randn(1, 1, 9000)),
                   ["x", "z_raw"], ["picker", "polarity", "detector"]))

# --- Edge-RED-PAN-Motion --------------------------------------------------
cfg = json.load(open(CK / "edge_rp90/config.json"))
sd = load_sd(CK / "edge_rp90/best.pt")
kw = {k: cfg[k] for k in ('input_size', 'block', 'polarity_head',
                          'polarity_output_channels', 'dropout_rate', 'pad_mode') if k in cfg}
if 'polarity_output_channels' not in kw and 'polarity.head.weight' in sd:
    kw['polarity_output_channels'] = int(sd['polarity.head.weight'].shape[0])
m = build_edge_rp90(**kw)
m.load_state_dict(sd)
meta.append(export("edge_rp90", Wrap3(m), (torch.randn(1, 3, 9000), torch.randn(1, 1, 9000)),
                   ["x", "z_raw"], ["picker", "polarity", "detector"]))

Path(OUT / "manifest.json").write_text(json.dumps(meta, indent=2))
print(f"\nmanifest -> {OUT/'manifest.json'}")
