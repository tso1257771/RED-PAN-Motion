"""Export EdgeRP90 (e189) to ONNX (fp32) and verify parity with PyTorch.
Fixed 90 s window (1,3,9000) + raw-Z (1,1,9000); outputs (picker, polarity, detector).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from pathlib import Path
from redpan_motion.inference.predictor import REDPANPredictor

CKPT = "checkpoints/edge_rp90/best.pt"   # epoch 189
OUT = Path("edge_model_design/deploy/edge_rp90_e189.onnx")
OUT.parent.mkdir(parents=True, exist_ok=True)


class Wrap(torch.nn.Module):
    """Explicit 2-input / 3-output wrapper for a clean ONNX signature."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, z_raw):
        picker, polarity, detector = self.m(x, z_raw=z_raw)
        return picker, polarity, detector


m = REDPANPredictor.from_checkpoint(CKPT, device="cpu").model.eval()
w = Wrap(m).eval()
x = torch.randn(1, 3, 9000)
z = torch.randn(1, 1, 9000)
with torch.no_grad():
    ref = w(x, z)

torch.onnx.export(
    w, (x, z), str(OUT),
    input_names=["x", "z_raw"],
    output_names=["picker", "polarity", "detector"],
    opset_version=17, do_constant_folding=True,
)
print(f"exported {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")

import onnx
onnx.checker.check_model(onnx.load(str(OUT)))
print("onnx.checker: OK")

import onnxruntime as ort
sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
out = sess.run(None, {"x": x.numpy(), "z_raw": z.numpy()})
print("parity (max |onnx - torch|):")
for i, name in enumerate(["picker", "polarity", "detector"]):
    d = float(np.abs(out[i] - ref[i].numpy()).max())
    print(f"  {name:9s} {d:.2e}   {'OK' if d < 1e-4 else 'MISMATCH'}")
