"""Convert a RED-PAN 60 s TensorFlow model (``TF60``) to a pure-torch
``Redpan60s`` checkpoint, with outputs verified to match.

A maintainer tool: ``redpan_motion/checkpoints/redpan_60s/`` already holds its output, so users
do not need to run it. It needs the TensorFlow model code and trained weights of
the original RED-PAN, https://github.com/tso1257771/RED-PAN (``redpan/legacy/
mtan_ARRU.py`` and ``pretrained_model/``); point ``REDPAN_TF_TOOLS`` and
``REDPAN_TF_DIR`` at them.

TF60 is the Keras ``unets.build_mtan_R2unet`` model (``mtan_ARRU.py``)
with ``input_size=(6000, 3)``, ``nb_filters=[6,12,18,24,30,36]`` — a 2-head
(picker + detector) MTAN R2U-Net, no polarity, 352,817 total params
(349,685 trainable + 3,132 BN running stats).

This reads the trained TF checkpoint (``train.hdf5`` / ``train.weights``), ports
every Conv1D / BatchNormalization into ``Redpan60s`` using the deterministic
Keras auto-name map (``Redpan60s.tf_layer_names``), and writes
``redpan_motion/checkpoints/redpan_60s/{best.pt, config.json}``. With ``--verify`` it runs both
models on CPU and reports the picker/detector max-abs difference (target < 1e-3;
observed ~2e-7).

TensorFlow is imported lazily and only here (CPU-only; the repo has no TF runtime
dependency). Run with ``CUDA_VISIBLE_DEVICES=""``: the conversion needs no GPU,
and CPU TensorFlow avoids cuDNN version mismatches.

Usage:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python scripts/convert_redpan_60s.py --verify
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from redpan_motion.models.redpan_60s import Redpan60s

DEFAULT_TF_TOOLS = os.environ.get("REDPAN_TF_TOOLS", "/path/to/RED-PAN")
DEFAULT_TF_DIR = os.environ.get(
    "REDPAN_TF_DIR", f"{DEFAULT_TF_TOOLS}/pretrained_model/<60 s model>")
DEFAULT_OUT = "redpan_motion/checkpoints/redpan_60s"


def load_tf_named_weights(tf_dir: str, tf_tools: str) -> dict:
    """Build the Keras TF60 model, load trained weights, return
    ``{f'{layer}|{var}': ndarray}`` for every Conv1D / BatchNormalization."""
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import sys
    sys.path.append(tf_tools)
    from REDPAN_tools.mtan_ARRU import unets  # noqa: E402

    frame = unets(input_size=(6000, 3))
    weights_file = os.path.join(tf_dir, "train.hdf5")
    if not os.path.exists(weights_file):
        weights_file = os.path.join(tf_dir, "train.weights")
    model = frame.build_mtan_R2unet(weights_file, input_size=(6000, 3))

    wd = {}
    for layer in model.layers:
        ws = layer.get_weights()
        if not ws:
            continue
        cn = layer.__class__.__name__
        if cn == "Conv1D":
            wd[f"{layer.name}|kernel"] = ws[0]
            wd[f"{layer.name}|bias"] = ws[1]
        elif cn == "BatchNormalization":
            wd[f"{layer.name}|gamma"] = ws[0]
            wd[f"{layer.name}|beta"] = ws[1]
            wd[f"{layer.name}|moving_mean"] = ws[2]
            wd[f"{layer.name}|moving_variance"] = ws[3]
        else:
            raise RuntimeError(f"unexpected weighted layer {cn}:{layer.name}")
    return wd, model


def port_weights(model: Redpan60s, tf_weights: dict) -> dict:
    """Port TF named weights into a ``Redpan60s`` state_dict. Conv kernels are
    transposed (K,In,Out)->(Out,In,K); BN maps gamma/beta/moving_* -> torch."""
    name_map = model.tf_layer_names()
    ref = model.state_dict()
    sd, used = {}, set()
    for path, tf in name_map.items():
        mod = model.get_submodule(path)
        if isinstance(mod, torch.nn.Conv1d):
            w = torch.tensor(tf_weights[f"{tf}|kernel"]).permute(2, 1, 0).contiguous()
            assert w.shape == ref[f"{path}.weight"].shape, (path, tf, w.shape)
            sd[f"{path}.weight"] = w
            sd[f"{path}.bias"] = torch.tensor(tf_weights[f"{tf}|bias"])
            used |= {f"{tf}|kernel", f"{tf}|bias"}
        elif isinstance(mod, torch.nn.BatchNorm1d):
            sd[f"{path}.weight"] = torch.tensor(tf_weights[f"{tf}|gamma"])
            sd[f"{path}.bias"] = torch.tensor(tf_weights[f"{tf}|beta"])
            sd[f"{path}.running_mean"] = torch.tensor(tf_weights[f"{tf}|moving_mean"])
            sd[f"{path}.running_var"] = torch.tensor(tf_weights[f"{tf}|moving_variance"])
            used |= {f"{tf}|{s}" for s in ("gamma", "beta", "moving_mean", "moving_variance")}
        else:
            raise TypeError(f"{path} -> {type(mod)}")
    for k, v in ref.items():
        if k.endswith("num_batches_tracked"):
            sd[k] = v
    missing = set(ref) - set(sd)
    extra = set(sd) - set(ref)
    unused = set(tf_weights) - used
    assert not missing and not extra, (list(missing)[:5], list(extra)[:5])
    assert not unused, f"unused TF tensors: {list(unused)[:5]}"
    return sd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tf-dir", default=DEFAULT_TF_DIR, help="dir with train.hdf5/train.weights")
    ap.add_argument("--tf-tools", default=DEFAULT_TF_TOOLS, help="dir containing REDPAN_tools/")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output checkpoint dir")
    ap.add_argument("--verify", action="store_true", help="run TF vs torch parity check")
    args = ap.parse_args()

    print("Loading TF60 weights from", args.tf_dir)
    tf_weights, tf_model = load_tf_named_weights(args.tf_dir, args.tf_tools)
    print(f"  extracted {len(tf_weights)} TF weight tensors")

    model = Redpan60s().eval()
    sd = port_weights(model, tf_weights)
    model.load_state_dict(sd)
    total = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for n, b in model.named_buffers()
                  if not n.endswith("num_batches_tracked"))
    print(f"  torch trainable params: {total}  (+{buffers} BN stats = {total + buffers})")
    assert total == 349685 and total + buffers == 352817

    max_pk = max_dt = None
    if args.verify:
        np.random.seed(42)
        x = np.random.randn(2, 6000, 3).astype(np.float32)
        tf_pk, tf_dt = tf_model.predict(x, verbose=0)
        with torch.no_grad():
            pk, dt = model(torch.tensor(x.transpose(0, 2, 1)))
        pk = pk.numpy().transpose(0, 2, 1)
        dt = dt.numpy().transpose(0, 2, 1)
        max_pk = float(np.abs(pk - tf_pk).max())
        max_dt = float(np.abs(dt - tf_dt).max())
        print(f"  PARITY  picker max-abs diff {max_pk:.3e} | detector {max_dt:.3e}  (target < 1e-3)")
        assert max_pk < 1e-3 and max_dt < 1e-3, "PARITY FAILED"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, out / "best.pt")
    config = {
        "_comment": (
            f"RED-PAN 60 s: a PyTorch port of the RED-PAN 60 s TensorFlow checkpoint "
            f"{Path(args.tf_dir).name} (architecture of Liao et al.), converted by "
            "scripts/convert_redpan_60s.py. Two heads, picker (P/S/N) and "
            "detector (event mask), no polarity. 349,685 trainable parameters "
            "plus 3,132 batch-norm statistics, 352,817 in total, equal to the "
            "TensorFlow count."),
        "_parity": {
            "picker_max_abs_diff": max_pk,
            "detector_max_abs_diff": max_dt,
            "note": "vs CPU TF, seed-42 random input; ~2e-7 (float32 noise). Run --verify to reproduce.",
        },
        "model_type": "redpan_60s",
        "input_size": [6000, 3],
        "nb_filters": [6, 12, 18, 24, 30, 36],
        "kernel_size": 7,
        "stride_size": 5,
        "upsize": 5,
        "rrconv_time": 3,
        "dropout_rate": 0.1,
        "picker_classes": 3,
        "detector_classes": 2,
        "source_tf_checkpoint": Path(args.tf_dir).name,
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Saved", out / "best.pt", "and", out / "config.json")


if __name__ == "__main__":
    main()
