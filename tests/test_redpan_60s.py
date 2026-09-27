"""Tests for Redpan60s — the pure-torch port of the original RED-PAN 60 s model.

Contract, param budget, length flexibility, and MAC budget. The TF-vs-torch
parity number (~2e-7) is verified by ``scripts/convert_redpan_60s.py --verify``
(needs a TF runtime + the trained checkpoint), not here.
"""
import os
import shutil

import pytest
import torch
import torch.nn as nn

from redpan_motion.inference import REDPANPredictor
from redpan_motion.checkpoints import checkpoint_path
from redpan_motion.models import build_redpan_60s, Redpan60s

TRAINABLE_PARAMS = 349_685
TOTAL_PARAMS = 352_817          # incl. BN running_mean/var (== TF count_params)
MAC_BUDGET = 0.09e9             # measured ~0.0823 GMAC per 6000-sample window
CKPT = str(checkpoint_path("redpan_60s"))


def test_param_count():
    m = Redpan60s()
    trainable = sum(p.numel() for p in m.parameters())
    stats = sum(b.numel() for n, b in m.named_buffers()
                if not n.endswith("num_batches_tracked"))
    assert trainable == TRAINABLE_PARAMS
    assert trainable + stats == TOTAL_PARAMS
    assert m.count_parameters() == TRAINABLE_PARAMS


def test_output_contract():
    m = Redpan60s().eval()
    with torch.no_grad():
        picker, detector = m(torch.randn(2, 3, 6000))
    assert picker.shape == (2, 3, 6000)
    assert detector.shape == (2, 2, 6000)
    # softmax over the class (channel) axis
    assert torch.allclose(picker.sum(1), torch.ones(2, 6000), atol=1e-4)
    assert torch.allclose(detector.sum(1), torch.ones(2, 6000), atol=1e-4)


@pytest.mark.parametrize("T", [3000, 6000, 9000])
def test_length_flexibility(T):
    m = Redpan60s().eval()
    with torch.no_grad():
        picker, detector = m(torch.randn(1, 3, T))
    assert picker.shape[-1] == T and detector.shape[-1] == T


def test_mac_budget():
    m = Redpan60s().eval()
    macs = [0]

    def hook(mod, inp, out):
        macs[0] += out.shape[-1] * mod.out_channels * (mod.in_channels // mod.groups) \
            * mod.weight.shape[2]

    hooks = [mod.register_forward_hook(hook)
             for mod in m.modules() if isinstance(mod, nn.Conv1d)]
    with torch.no_grad():
        m(torch.zeros(1, 3, 6000))
    for h in hooks:
        h.remove()
    assert macs[0] < MAC_BUDGET, f"{macs[0]/1e9:.4f} GMAC exceeds budget"


@pytest.mark.skipif(not os.path.exists(CKPT), reason="ported checkpoint not present")
def test_checkpoint_loads():
    m = build_redpan_60s(pretrained_weights=CKPT).eval()
    assert m.count_parameters() == TRAINABLE_PARAMS
    with torch.no_grad():
        picker, detector = m(torch.randn(1, 3, 6000))
    assert picker.shape == (1, 3, 6000) and detector.shape == (1, 2, 6000)


@pytest.mark.skipif(not os.path.exists(CKPT), reason="ported checkpoint not present")
def test_from_checkpoint_dispatch():
    """The documented entry point must build Redpan60s, not the base MTAN.

    Regression guard. The model_type dispatch in from_checkpoint once had
    branches only for 'edge_rp90_v1' and 'mtan_r2unet_rp90_motion*' and then
    fell through to MTAN_R2UNet, so this checkpoint's config declaring
    model_type='redpan_60s' silently built the WRONG architecture and
    load_state_dict raised on several hundred missing keys — the shipped 60 s
    model was unloadable through from_checkpoint while every other family
    loaded fine.
    """
    pred = REDPANPredictor.from_checkpoint(CKPT, device="cpu")
    assert isinstance(pred.model, Redpan60s)
    # native 60 s window, not the 90 s models' default
    assert pred.pred_npts == 6000
    with torch.no_grad():
        picker, detector = pred.model.eval()(torch.randn(1, 3, 6000))
    assert picker.shape == (1, 3, 6000) and detector.shape == (1, 2, 6000)


@pytest.mark.skipif(not os.path.exists(CKPT), reason="ported checkpoint not present")
def test_from_checkpoint_autodetect_without_config(tmp_path):
    """With no sibling config.json, the state_dict keys alone must identify it.

    The 60 s port is a conversion of the TF graph, so its key namespace
    (ps_init/m_init/pred_ps/...) is disjoint from the MTAN line's — that is
    what the auto-detect keys off when no model_type is available anywhere.
    """
    shutil.copy(CKPT, tmp_path / "best.pt")
    pred = REDPANPredictor.from_checkpoint(str(tmp_path / "best.pt"), device="cpu")
    assert isinstance(pred.model, Redpan60s)
