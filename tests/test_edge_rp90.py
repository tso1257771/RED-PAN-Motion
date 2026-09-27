"""Tests for EdgeRP90 (edge_rp90_v1): output contract, length flexibility, and the
parameter and MAC budgets the architecture was designed to."""
import torch
import torch.nn as nn

from redpan_motion.models import build_edge_rp90, EdgeRP90

PARAM_BUDGET = 320_000
MAC_BUDGET = 0.22e9          # MACs per 9000-sample window
FULLRES_SHARE_MAX = 0.45     # fraction of MACs at T >= input length


def _forward(model, B=2, T=9000):
    x = torch.randn(B, 3, T)
    z = torch.randn(B, 1, T)
    with torch.no_grad():
        return model(x, z_raw=z)


def _count_macs(model, T=9000):
    """Hook-count Conv1d/Linear MACs for one window; also the share at full (>=T)
    temporal resolution."""
    tot = {"all": 0.0, "fullres": 0.0}
    hooks = []

    def make(m):
        def hook(mod, inp, out):
            if isinstance(mod, nn.Conv1d):
                out_len = out.shape[-1]
                mac = out_len * mod.out_channels * (mod.in_channels // mod.groups) \
                    * mod.kernel_size[0]
                is_full = out_len >= T
            elif isinstance(mod, nn.Linear):
                mac = mod.in_features * mod.out_features
                is_full = False
            else:
                return
            tot["all"] += mac
            if is_full:
                tot["fullres"] += mac
        return hook

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            hooks.append(m.register_forward_hook(make(m)))
    x = torch.randn(1, 3, T)
    z = torch.randn(1, 1, T)
    with torch.no_grad():
        model(x, z_raw=z)
    for h in hooks:
        h.remove()
    return tot["all"], tot["fullres"]


def test_output_contract():
    m = build_edge_rp90().eval()
    picker, polarity, detector = _forward(m)
    assert picker.shape == (2, 3, 9000)
    assert polarity.shape == (2, 3, 9000)
    assert detector.shape == (2, 2, 9000)
    for name, t in (("picker", picker), ("polarity", polarity), ("detector", detector)):
        s = t.sum(dim=1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-4), f"{name} not a softmax"


def test_falls_back_to_x_when_no_zraw():
    m = build_edge_rp90().eval()
    x = torch.randn(2, 3, 9000)
    with torch.no_grad():
        picker, polarity, detector = m(x)  # no z_raw
    assert polarity.shape == (2, 3, 9000)


def test_arbitrary_length_roundtrip():
    m = build_edge_rp90().eval()
    for T in (6001, 3000, 12345):
        picker, polarity, detector = _forward(m, B=1, T=T)
        assert picker.shape == (1, 3, T)
        assert polarity.shape == (1, 3, T)
        assert detector.shape == (1, 2, T)


def test_param_budget():
    n = build_edge_rp90().count_parameters()
    assert n <= PARAM_BUDGET, f"{n:,} params exceeds budget {PARAM_BUDGET:,}"


def test_mac_budget():
    macs, fullres = _count_macs(build_edge_rp90().eval())
    assert macs <= MAC_BUDGET, f"{macs/1e9:.4f} GMAC exceeds {MAC_BUDGET/1e9:.2f}"
    share = fullres / macs
    assert share <= FULLRES_SHARE_MAX, f"full-res share {share:.2%} exceeds {FULLRES_SHARE_MAX:.0%}"


def test_dw_variant_builds_same_contract():
    m = build_edge_rp90(block="dw").eval()
    picker, polarity, detector = _forward(m)
    assert picker.shape == (2, 3, 9000)
    assert detector.shape == (2, 2, 9000)


def test_fullres_sep_polarity_flag():
    m = build_edge_rp90(polarity_head="fullres_sep").eval()
    _, polarity, _ = _forward(m)
    assert polarity.shape == (2, 3, 9000)


def test_all_params_receive_grad():
    # DDP (default, no find_unused_parameters) aborts if any parameter does not
    # participate in the loss. Every trainable param must get a gradient from the
    # three outputs (regression guard for the removed early-exit gate head).
    m = build_edge_rp90().train()
    x = torch.randn(2, 3, 9000)
    z = torch.randn(2, 1, 9000)
    picker, polarity, detector = m(x, z_raw=z)
    (picker.mean() + polarity.mean() + detector.mean()).backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"params with no grad (would break DDP): {missing}"
