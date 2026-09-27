#!/usr/bin/env python
"""Smoke test: prove the standalone redpan_motion package imports, builds the
production model from each shipped checkpoint's config, strict-loads the weights,
and runs a forward pass — all without importing TensorFlow.

Run from the project root:
    PYTHONPATH=. python scripts/verify_install.py
"""
import inspect
import json
import sys

import torch

from redpan_motion.checkpoints import CHECKPOINT_DIR as CKPT


def fail(msg):
    print(f"  FAIL: {msg}")
    sys.exit(1)


def main():
    # 1. Imports (must not pull TensorFlow)
    import redpan_motion
    from redpan_motion.models import build_mtan_r2unet_rp90_motion
    from redpan_motion.inference import REDPANPredictor  # noqa: F401
    from redpan_motion.training import REDPANTrainer  # noqa: F401

    assert "tensorflow" not in sys.modules, "TensorFlow imported at package import time!"
    print(f"[1] import redpan_motion v{redpan_motion.__version__} — OK (no tensorflow in sys.modules)")

    # 2. Build each checkpoint from its own config, strict-load, forward.
    dirs = sorted(CKPT.glob("redpan_motion"))
    if not dirs:
        fail("no checkpoints found")
    sig = inspect.signature(build_mtan_r2unet_rp90_motion)
    for d in dirs:
        cfg = json.load(open(d / "config.json"))
        kwargs = {k: v for k, v in cfg.items() if k in sig.parameters}
        if "input_size" in kwargs:
            kwargs["input_size"] = tuple(kwargs["input_size"])
        model = build_mtan_r2unet_rp90_motion(**kwargs)
        state = torch.load(d / "best.pt", map_location="cpu")
        state = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state, strict=True)   # must match exactly
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            picker, polarity, detector = model(torch.randn(1, 3, cfg["input_size"][0]))
        print(f"[2] {d.name}: {n_params/1e3:.0f}k params, forward -> "
              f"picker {tuple(picker.shape)}, polarity {tuple(polarity.shape)}, "
              f"detector {tuple(detector.shape)} — OK")

    # 3. High-level predictor (sliding-window, from_checkpoint).
    latest = dirs[-1] / "best.pt"
    pred = REDPANPredictor.from_checkpoint(str(latest), device="cpu")
    print(f"[3] REDPANPredictor.from_checkpoint({dirs[-1].name}) — OK ({type(pred.model).__name__})")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
