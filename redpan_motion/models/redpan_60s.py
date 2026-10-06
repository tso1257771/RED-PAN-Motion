"""
Redpan60s — pure-PyTorch port of the RED-PAN 60 s architecture (Liao et al.),
with weights converted from a released TensorFlow checkpoint (output parity ~2e-7).
The shipped weights are REDPAN_60s_240107 (release 2024-01-07, trained with
Romanian data), not the model evaluated in the 2022 paper.

The original ("TF60") is an MTAN R2U-Net built by
``REDPAN_tools/mtan_ARRU.py::unets.build_mtan_R2unet`` with
``input_size=(6000, 3)``, ``nb_filters=[6, 12, 18, 24, 30, 36]``, ``depth=6``,
``kernel_size=7``, ``stride_size=upsize=5``, ``RRconv_time=3``. Two heads, **no
polarity**:

  - Picker:   (B, 3, T) — P / S / Noise           (softmax)
  - Detector: (B, 2, T) — event / no-event mask    (softmax)

Faithful-port note (important): the TF builder *instantiates* a full dual-MTAN
(picker + detector) attention block at every encoder and decoder level, but the
Keras functional ``Model(inputs, [picker, detector])`` **prunes every layer not
on the path to the two outputs**. Only ``pred_PS``/``pred_mask`` (the *last*
decoder MTAN, ``*_mtan_D5``) feed the outputs, and that last block references the
*first* MTAN (``*_mtan_init``). So the trained/alive network keeps MTAN attention
**only at the input level and the final decoder level** — every intermediate
MTAN block and the extra bottleneck RRConv are dead weight that Keras never saves.
This module replicates the *alive* subgraph exactly (105 Conv1D + 86 BatchNorm =
352,817 params), which is what the checkpoint contains.

Keras parity details baked in here:
  * Conv1D ``padding="same"`` → asymmetric TF/SAME padding (matters for the
    stride-5 down-convs and the ``UpSampling1D`` crops); see ``SameConv1d``.
  * BatchNormalization ``epsilon=1e-3`` (Keras default), not torch's 1e-5.
  * ``UpSampling1D`` = nearest repeat; decoder concat order is ``[skip, up]``.
  * softmax over the channel axis (TF axis=-1).

Weights are ported by ``scripts/convert_redpan_60s.py`` (TF → torch), which
uses the deterministic Keras auto-name map produced by ``tf_layer_names()``.
"""
from __future__ import annotations

import logging
import math
import pickle
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

BN_EPS = 1e-3  # Keras BatchNormalization default epsilon


def _keras_same_pad(length: int, kernel: int, stride: int) -> Tuple[int, int]:
    """TF/Keras ``padding='same'`` left/right pad for a 1-D conv."""
    out = math.ceil(length / stride)
    pad = max((out - 1) * stride + kernel - length, 0)
    return pad // 2, pad - pad // 2


class SameConv1d(nn.Conv1d):
    """``nn.Conv1d`` with TF/Keras ``padding='same'`` semantics (dynamic,
    stride-aware, asymmetric). Standard weight/bias params so it profiles and
    serialises like any Conv1d."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1):
        super().__init__(in_ch, out_ch, kernel, stride=stride, padding=0, bias=True)
        self._k = kernel
        self._s = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._k > 1:
            pl, pr = _keras_same_pad(x.shape[-1], self._k, self._s)
            x = F.pad(x, (pl, pr))
        return F.conv1d(x, self.weight, self.bias, self._s, 0, self.dilation, self.groups)


class ConvBNAct(nn.Module):
    """TF ``conv_unit``: Conv1D(SAME) → BN(eps=1e-3) → ReLU → Dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 1,
                 dropout: float = 0.1, act: bool = True):
        super().__init__()
        self.conv = SameConv1d(in_ch, out_ch, kernel, stride)
        self.bn = nn.BatchNorm1d(out_ch, eps=BN_EPS)
        self.act = act
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        if self.act:
            x = F.relu(x)
        return self.drop(x)


class RRConv(nn.Module):
    """TF ``RRconv_unit`` (Recurrent-Residual conv, RRconv_time=3).

    ``u = conv_unit(x)`` (carries the block stride); ``sc = conv1d(u)`` (plain
    SAME conv, no BN/act — the residual shortcut); then 3 recurrent conv_units
    with a shared ``+u`` residual; return ``r + sc``. Sub-layer *construction*
    order (init, shortcut, loop×3) mirrors Keras' layer numbering.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 1,
                 dropout: float = 0.1, iters: int = 3):
        super().__init__()
        self.iters = iters
        self.init_conv = ConvBNAct(in_ch, out_ch, kernel, stride, dropout)
        self.shortcut = SameConv1d(out_ch, out_ch, kernel, 1)  # no BN/act
        self.loops = nn.ModuleList(
            [ConvBNAct(out_ch, out_ch, kernel, 1, dropout) for _ in range(iters)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.init_conv(x)
        sc = self.shortcut(u)
        r = u
        for i in range(self.iters):
            r = self.loops[i](r + u)
        return r + sc


def _nearest_upsample(x: torch.Tensor, k: int) -> torch.Tensor:
    """Keras ``UpSampling1D`` (nearest repeat) on ``(B, C, T)``.

    Elementwise-identical to ``x.repeat_interleave(k, dim=2)``, but exports to
    ONNX as Expand+Reshape. ``repeat_interleave`` traces into a dynamic ``Loop``
    over a tensor sequence, and the 8 upsample sites then dominate the runtime
    of the exported graph on a CPU. Torch-side output is bit identical, so the
    TF parity of this port is unaffected.
    """
    b, c, t = x.shape
    return x.unsqueeze(-1).expand(b, c, t, k).reshape(b, c, t * k)


class UpConv(nn.Module):
    """TF ``upconv_unit`` (no attention path): UpSampling1D(x upsize) →
    Conv(SAME) → BN → ReLU → Dropout → crop-to-skip → concat([skip, up])."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, upsize: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.upsize = upsize
        self.conv = SameConv1d(in_ch, out_ch, kernel, 1)
        self.bn = nn.BatchNorm1d(out_ch, eps=BN_EPS)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        u = _nearest_upsample(x, self.upsize)                  # UpSampling1D (nearest)
        u = self.drop(F.relu(self.bn(self.conv(u))))
        diff = u.shape[-1] - skip.shape[-1]
        if diff > 0:                                           # Keras Cropping1D
            u = u[:, :, diff // 2: diff // 2 + skip.shape[-1]]
        elif diff < 0:
            u = F.pad(u, (0, -diff))
        return torch.cat([skip, u], dim=1)                     # concat([skip, up])


class MTANDown(nn.Module):
    """TF ``mtan_att_block(mode='down')``: attention gate + a strided conv_unit.

    ``x = concat([pre_att, pre_target])`` → 1×1 conv → BN → ReLU → 1×1 conv → BN
    → sigmoid → gate ``target`` → ``conv_unit(stride)``.
    """

    def __init__(self, cat_in_ch: int, nb_in: int, target_ch: int, nb_out: int,
                 stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.att1 = SameConv1d(cat_in_ch, nb_in, 1)
        self.bn1 = nn.BatchNorm1d(nb_in, eps=BN_EPS)
        self.att2 = SameConv1d(nb_in, nb_in, 1)
        self.bn2 = nn.BatchNorm1d(nb_in, eps=BN_EPS)
        self.out_conv = ConvBNAct(target_ch, nb_out, 7, stride, dropout)

    def forward(self, pre_att: torch.Tensor, pre_target: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pre_att, pre_target], dim=1)
        a = F.relu(self.bn1(self.att1(x)))
        a = torch.sigmoid(self.bn2(self.att2(a)))
        return self.out_conv(a * target)


class MTANUp(nn.Module):
    """TF ``mtan_att_block(mode='up')``: upconv the previous attention, gate the
    decoder feature. ``x = upconv(pre_att, skip=pre_target)`` → 1×1 → BN → ReLU →
    1×1 → BN → sigmoid → multiply with ``target``."""

    def __init__(self, pre_att_ch: int, nb_out: int, cat_in_ch: int, nb_in: int,
                 upsize: int = 5, dropout: float = 0.1):
        super().__init__()
        self.up = UpConv(pre_att_ch, nb_out, 7, upsize, dropout)
        self.att1 = SameConv1d(cat_in_ch, nb_in, 1)
        self.bn1 = nn.BatchNorm1d(nb_in, eps=BN_EPS)
        self.att2 = SameConv1d(nb_in, nb_in, 1)
        self.bn2 = nn.BatchNorm1d(nb_in, eps=BN_EPS)

    def forward(self, pre_att: torch.Tensor, pre_target: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        x = self.up(pre_att, pre_target)
        a = F.relu(self.bn1(self.att1(x)))
        a = torch.sigmoid(self.bn2(self.att2(a)))
        return a * target


class Redpan60s(nn.Module):
    """Pure-torch port of the RED-PAN 60 s architecture (the *alive* subgraph).

    Output contract: ``forward(x) -> (picker, detector)`` with
    ``x`` shape ``(B, 3, T)`` (T=6000 for the trained model; length-flexible).
    """

    NB = [6, 12, 18, 24, 30, 36]

    def __init__(self, input_size: Tuple[int, int] = (6000, 3),
                 nb_filters: Optional[List[int]] = None, kernel_size: int = 7,
                 stride_size: int = 5, upsize: int = 5, dropout_rate: float = 0.1,
                 rrconv_time: int = 3, picker_classes: int = 3,
                 detector_classes: int = 2):
        super().__init__()
        F_ = nb_filters or self.NB
        self.nb_filters = list(F_)
        in_ch = input_size[1]
        k, s, up, dr, it = kernel_size, stride_size, upsize, dropout_rate, rrconv_time

        # ---- input level ----
        self.rr_e0 = RRConv(in_ch, F_[0], k, 1, dr, it)
        self.ps_init = MTANDown(F_[0] * 2, F_[0], F_[0], F_[0], 1, dr)
        self.m_init = MTANDown(F_[0] * 2, F_[0], F_[0], F_[0], 1, dr)

        # ---- encoder backbone (alive; intermediate MTANs pruned) ----
        self.rr_exp = nn.ModuleList()
        self.rr_down = nn.ModuleList()
        for i in range(len(F_) - 1):
            self.rr_exp.append(RRConv(F_[i], F_[i], k, 1, dr, it))
            self.rr_down.append(RRConv(F_[i], F_[i + 1], k, s, dr, it))

        # ---- decoder backbone ----
        self.up = nn.ModuleList()
        self.df = nn.ModuleList()
        for i in range(len(F_)):
            skip_ch = F_[-1 - i]              # Es[-1-i]
            out_ch = F_[-1 - i]
            in_ch_up = F_[-1] if i == 0 else F_[-i]   # Es[-1] or previous D_fus
            self.up.append(UpConv(in_ch_up, out_ch, k, up, dr))
            self.df.append(RRConv(skip_ch + out_ch, out_ch, k, 1, dr, it))

        # ---- final-level MTAN (the only decoder MTAN that survives pruning) ----
        # pre_att = *_mtan_init (F_[0]); skip/pre_target = D5 (= skip_ch F_[0] + out F_[0]);
        # target = df5 (F_[0]).
        d5_cat = self.nb_filters[0] * 2                 # D5 channel count (12)
        self.ps_d5 = MTANUp(F_[0], F_[0], d5_cat + F_[0], F_[0], up, dr)
        self.m_d5 = MTANUp(F_[0], F_[0], d5_cat + F_[0], F_[0], up, dr)

        # ---- output heads (1x1 conv → softmax) ----
        self.pred_ps = SameConv1d(F_[0], picker_classes, 1)
        self.pred_mask = SameConv1d(F_[0], detector_classes, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        F_ = self.nb_filters
        e0 = self.rr_e0(x)
        ps_att = self.ps_init(e0, e0, e0)
        m_att = self.m_init(e0, e0, e0)

        # encoder backbone; keep E0..E5 for skips
        es = [e0]
        cur = e0
        for i in range(len(F_) - 1):
            exp = self.rr_exp[i](cur)
            cur = self.rr_down[i](exp)
            es.append(cur)

        # decoder backbone
        d = es[-1]
        d_up = None
        d_fus = None
        for i in range(len(F_)):
            skip = es[-1 - i]
            d_up = self.up[i](d, skip)      # "D" (pre_target for final MTAN)
            d_fus = self.df[i](d_up)        # "D_fus" (target for final MTAN)
            d = d_fus

        # final-level attention → heads
        ps_out = self.ps_d5(ps_att, d_up, d_fus)
        m_out = self.m_d5(m_att, d_up, d_fus)
        picker = F.softmax(self.pred_ps(ps_out).float(), dim=1)
        detector = F.softmax(self.pred_mask(m_out).float(), dim=1)
        return picker, detector

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Deterministic Keras auto-name map (for the TF→torch weight port).
    # Replays build_mtan_R2unet's *instantiation* order — including the layers
    # Keras later prunes — so the global conv/bn counters land on the exact
    # ``conv1d_N`` / ``batch_normalization_N`` names the checkpoint uses.
    # ------------------------------------------------------------------
    def tf_layer_names(self) -> dict:
        """Return ``{torch_submodule_path: tf_layer_name}`` for every Conv1d and
        BatchNorm1d, keyed by module path (e.g. ``rr_e0.init_conv.conv``)."""
        gen = _KerasNameGen()
        F_ = self.nb_filters
        m: dict = {}

        def rrconv(prefix: str):
            m[f"{prefix}.init_conv.conv"] = gen.conv()
            m[f"{prefix}.init_conv.bn"] = gen.bn()
            m[f"{prefix}.shortcut"] = gen.conv()
            for j in range(3):
                m[f"{prefix}.loops.{j}.conv"] = gen.conv()
                m[f"{prefix}.loops.{j}.bn"] = gen.bn()

        def mtan_down(prefix: Optional[str]):
            # prefix None -> dead block, just advance counters
            n = [gen.conv(), gen.bn(), gen.conv(), gen.bn(), gen.conv(), gen.bn()]
            if prefix is not None:
                m[f"{prefix}.att1"] = n[0]
                m[f"{prefix}.bn1"] = n[1]
                m[f"{prefix}.att2"] = n[2]
                m[f"{prefix}.bn2"] = n[3]
                m[f"{prefix}.out_conv.conv"] = n[4]
                m[f"{prefix}.out_conv.bn"] = n[5]

        def mtan_up(prefix: Optional[str]):
            n = [gen.conv(), gen.bn(), gen.conv(), gen.bn(), gen.conv(), gen.bn()]
            if prefix is not None:
                m[f"{prefix}.up.conv"] = n[0]
                m[f"{prefix}.up.bn"] = n[1]
                m[f"{prefix}.att1"] = n[2]
                m[f"{prefix}.bn1"] = n[3]
                m[f"{prefix}.att2"] = n[4]
                m[f"{prefix}.bn2"] = n[5]

        def rrconv_dead():
            # advance counters for a pruned RRConv (5 conv + 4 bn)
            gen.conv()
            gen.bn()
            gen.conv()
            for _ in range(3):
                gen.conv()
                gen.bn()

        # --- instantiation order (mirrors build_mtan_R2unet exactly) ---
        rrconv("rr_e0")
        mtan_down("ps_init")
        mtan_down("m_init")
        for i in range(len(F_) - 1):
            rrconv(f"rr_exp.{i}")
            mtan_down(None)            # PS_mtan_E{i} (pruned)
            mtan_down(None)            # M_mtan_E{i}  (pruned)
            rrconv(f"rr_down.{i}")
            if i == len(F_) - 2:
                rrconv_dead()          # bottleneck exp_E6 (pruned)
        for i in range(len(F_)):
            # decoder upconv_unit -> 1 conv + 1 bn
            m[f"up.{i}.conv"] = gen.conv()
            m[f"up.{i}.bn"] = gen.bn()
            rrconv(f"df.{i}")
            if i < len(F_) - 1:
                mtan_up(None)          # PS_mtan_D{i} (pruned)
                mtan_up(None)          # M_mtan_D{i}  (pruned)
            else:
                mtan_up("ps_d5")
                mtan_up("m_d5")
        # explicit-named output heads
        m["pred_ps"] = "pred_PS"
        m["pred_mask"] = "pred_mask"
        return m


class _KerasNameGen:
    """Reproduces Keras' global per-class auto-naming counters."""

    def __init__(self):
        self._c = 0
        self._b = 0

    def conv(self) -> str:
        n = "conv1d" if self._c == 0 else f"conv1d_{self._c}"
        self._c += 1
        return n

    def bn(self) -> str:
        n = "batch_normalization" if self._b == 0 else f"batch_normalization_{self._b}"
        self._b += 1
        return n


def build_redpan_60s(input_size: Tuple[int, int] = (6000, 3),
                      pretrained_weights: Optional[str] = None,
                      **_ignored) -> Redpan60s:
    """Build the TF60 torch model. ``pretrained_weights`` loads a torch
    state_dict (``best.pt`` produced by ``scripts/convert_redpan_60s.py``)."""
    model = Redpan60s(input_size=input_size)
    if pretrained_weights is not None:
        try:
            raw = torch.load(pretrained_weights, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, RuntimeError) as exc:
            logger.warning("weights_only load failed (%s); retrying weights_only=False", exc)
            raw = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
        model.load_state_dict(sd)
    return model
