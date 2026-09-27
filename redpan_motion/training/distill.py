"""
Distillation losses for training EdgeRP90 from larger teacher models.

Not used to train the shipped ``edge_rp90`` checkpoint, which was trained with the
supervised recipe alone. This module wraps ``MultiTaskLossWithPolarity``:

  L = L_sup  +  kd_w · L_KD  +  fa_w · L_false_alarm  +  gate_w · L_gate  [+ hint]

  - **Multi-teacher KD** from frozen teacher checkpoints. The detector target is a
    weighted ensemble; picker and polarity targets are equal averages. Temperature
    is recovered from the teachers' softmax probabilities via softmax(log p / T),
    exact because the teachers' forward returns post-softmax probabilities.
  - **Noise clamp**: on labelled noise traces the teacher is never trusted
    positively. KD is disabled there, and the hard [0,0,1] / [0,1] targets of the
    supervised loss decide.
  - **Double S-KD** on traces with 10-30 s S-P, read from the picker targets, so no
    dataset change is needed.
  - **L_false_alarm**: extra cross-entropy on the highest-risk samples of the noise
    traces already in each batch.
  - **L_gate**: trains the early-exit gate head.

The FitNets feature hints are not wired to the models: ``hint_weight`` is accepted
and off by default.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_EPS = 1e-8


def load_teachers(checkpoint_paths: Sequence[str], device: str = "cuda"):
    """Load frozen teacher models (v49 / large / xlarge) for distillation.

    Uses ``REDPANPredictor.from_checkpoint`` so each teacher rebuilds its exact
    architecture (different nb_filters per ladder rung) from its sibling config.
    Returns a list of eval-mode, gradient-frozen ``nn.Module`` teachers.
    """
    from redpan_motion.inference.predictor import REDPANPredictor
    teachers = []
    for path in checkpoint_paths:
        model = REDPANPredictor.from_checkpoint(path, device=device).model
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        teachers.append(model)
        logger.info("Loaded teacher: %s (%d params)", path,
                    sum(p.numel() for p in model.parameters()))
    return teachers


def _temp_dist(prob: torch.Tensor, T: float) -> torch.Tensor:
    """Temperature-scaled distribution recovered from a probability tensor:
    softmax(log(p)/T) == softmax(logit/T) (the additive constant cancels)."""
    return F.softmax(torch.log(prob.clamp_min(_EPS)) / T, dim=1)


def _temp_logdist(prob: torch.Tensor, T: float) -> torch.Tensor:
    return F.log_softmax(torch.log(prob.clamp_min(_EPS)) / T, dim=1)


def _ensemble(probs: List[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    """Weighted average of teacher probability tensors (weights renormalized)."""
    w = torch.tensor(weights[:len(probs)], dtype=probs[0].dtype, device=probs[0].device)
    w = w / w.sum()
    out = probs[0] * w[0]
    for i in range(1, len(probs)):
        out = out + probs[i] * w[i]
    return out


class DistillLossWrapper(nn.Module):
    """Wrap a base ``MultiTaskLossWithPolarity`` with the EdgeRP90 distillation recipe.

    Call ``forward(student_out, teacher_outs, pick_t, pol_t, det_t, pol_mask, ...)``.
    ``student_out`` = ``(picker, polarity, detector[, gate])`` (softmax probs; gate is
    a (B,1) logit if the student ran with ``return_aux=True``). ``teacher_outs`` = list
    of ``(picker, polarity, detector)`` softmax-prob tuples from the frozen teachers.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        kd_weight: float = 0.40,
        kd_weight_final: float = 0.15,
        fa_weight: float = 0.50,
        gate_weight: float = 0.05,
        hint_weight: float = 0.0,
        T_picker: float = 2.0,
        T_detector: float = 2.0,
        T_polarity: float = 1.5,
        detector_teacher_weights: Sequence[float] = (0.6, 0.2, 0.2),
        s_kd_boost: float = 2.0,
        s_p_lo_sec: float = 10.0,
        s_p_hi_sec: float = 30.0,
        sample_rate: float = 100.0,
        fa_topk: int = 128,
        fa_ce_scale: float = 5.0,
    ):
        super().__init__()
        self.base = base_loss
        self.kd_weight = kd_weight
        self.kd_weight_final = kd_weight_final
        self.fa_weight = fa_weight
        self.gate_weight = gate_weight
        self.hint_weight = hint_weight
        self.T_picker, self.T_detector, self.T_polarity = T_picker, T_detector, T_polarity
        self.det_w = tuple(detector_teacher_weights)
        self.s_kd_boost = s_kd_boost
        self.s_p_lo = int(s_p_lo_sec * sample_rate)
        self.s_p_hi = int(s_p_hi_sec * sample_rate)
        self.fa_topk = fa_topk
        self.fa_ce_scale = fa_ce_scale
        # Match the base loss's reduction so KD and supervised terms are on one scale.
        self._sum_over_t = getattr(base_loss, "loss_reduction", "sum_over_batch") \
            == "sum_over_batch"

    def kd_weight_at(self, epoch_frac: float) -> float:
        """Linearly decay the KD weight from kd_weight → kd_weight_final."""
        f = min(max(epoch_frac, 0.0), 1.0)
        return self.kd_weight + (self.kd_weight_final - self.kd_weight) * f

    def _reduce(self, per_bt: torch.Tensor, keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reduce a (B, T) per-sample loss to a scalar, matching the base reduction.
        ``keep`` is an optional (B,) mask of traces to include (mean over kept)."""
        per_b = per_bt.sum(dim=-1) if self._sum_over_t else per_bt.mean(dim=-1)
        if keep is not None:
            denom = keep.sum().clamp_min(1.0)
            return (per_b * keep).sum() / denom
        return per_b.mean()

    def _kd_channels(self, s_prob, t_prob, T):
        """Per-(B, class, T) KD contributions (KL, teacher||student) at temperature T."""
        s_log = _temp_logdist(s_prob, T)
        t = _temp_dist(t_prob, T)
        # KL(t || s) per element = t * (log t - log s); keep the class axis for weighting.
        return (t * (torch.log(t.clamp_min(_EPS)) - s_log)) * (T * T)

    def forward(
        self,
        student_out: Tuple[torch.Tensor, ...],
        teacher_outs: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        picker_target, polarity_target, detector_target, polarity_mask,
        epoch_frac: float = 0.0,
        return_individual: bool = False,
    ):
        picker_s, polarity_s, detector_s = student_out[0], student_out[1], student_out[2]
        gate_logit = student_out[3] if len(student_out) > 3 else None
        device = picker_s.device

        # ── Supervised base loss (hard targets; rules on noise) ──
        base = self.base(
            picker_s, polarity_s, detector_s,
            picker_target, polarity_target, detector_target,
            polarity_mask=polarity_mask, return_individual=True)
        total, picker_loss, polarity_loss, detector_loss = base[0], base[1], base[2], base[-1]

        # ── Per-trace flags ──
        # Event channel of the detector target (channel 0) is the labeled event mask.
        has_event = (detector_target[:, 0, :].amax(dim=-1) > 0.5).float()   # (B,)
        is_noise = 1.0 - has_event
        # S-P (in samples) from the picker Gaussian-peak targets [P, S, N].
        p_pos = picker_target[:, 0, :].argmax(dim=-1)
        s_pos = picker_target[:, 1, :].argmax(dim=-1)
        s_minus_p = (s_pos - p_pos).float()
        long_sp = ((s_minus_p >= self.s_p_lo) & (s_minus_p <= self.s_p_hi)
                   & (has_event > 0.5)).float()                              # (B,)

        # ── Multi-teacher soft targets ──
        t_pick = [t[0] for t in teacher_outs]
        t_pol = [t[1] for t in teacher_outs]
        t_det = [t[2] for t in teacher_outs]
        pick_soft = _ensemble(t_pick, [1.0] * len(t_pick))            # equal
        pol_soft = _ensemble(t_pol, [1.0] * len(t_pol))              # equal
        det_soft = _ensemble(t_det, self.det_w)                      # v49-anchored

        # KD is trusted ONLY on event traces (noise clamp): keep = has_event.
        keep = has_event

        # Picker KD, with the S channel (index 1) boosted on long-S-P traces.
        pick_kd_c = self._kd_channels(picker_s, pick_soft, self.T_picker)   # (B,3,T)
        s_boost = 1.0 + (self.s_kd_boost - 1.0) * long_sp                    # (B,)
        chan_w = torch.ones(3, device=device)
        pick_kd = (pick_kd_c[:, 0, :] * chan_w[0]
                   + pick_kd_c[:, 1, :] * s_boost.unsqueeze(-1)
                   + pick_kd_c[:, 2, :] * chan_w[2])                         # (B,T)
        picker_kd = self._reduce(pick_kd, keep)

        det_kd = self._kd_channels(detector_s, det_soft, self.T_detector).sum(dim=1)
        detector_kd = self._reduce(det_kd, keep)

        # Polarity KD only where the P-labeled polarity mask is active (and on events).
        pmask = polarity_mask
        if pmask.dim() == 3:
            pmask = pmask.abs().amax(dim=1)                                  # (B,T)
        pol_kd = self._kd_channels(polarity_s, pol_soft, self.T_polarity).sum(dim=1)
        pol_kd = pol_kd * (pmask > 0).float()
        polarity_kd = self._reduce(pol_kd, keep)

        kd_loss = picker_kd + detector_kd + polarity_kd

        # ── L_false_alarm: extra CE on the highest-risk samples of noise traces ──
        # risk = max student event/P/S probability (the would-be false trigger).
        risk = torch.maximum(detector_s[:, 0, :],
                             torch.maximum(picker_s[:, 0, :], picker_s[:, 1, :]))  # (B,T)
        k = min(self.fa_topk, risk.shape[-1])
        topk_vals, _ = risk.topk(k, dim=-1)                                  # (B,k)
        # penalize event-probability mass on noise: -log(1 - p) toward background.
        fa_per_b = (-torch.log((1.0 - topk_vals).clamp_min(_EPS))).sum(dim=-1)  # (B,)
        denom = is_noise.sum().clamp_min(1.0)
        fa_loss = self.fa_ce_scale * (fa_per_b * is_noise).sum() / denom

        # ── L_gate: window-eventness BCE (event=1, noise=0) ──
        if gate_logit is not None:
            gate_loss = F.binary_cross_entropy_with_logits(
                gate_logit.squeeze(-1), has_event)
        else:
            gate_loss = torch.zeros((), device=device)

        kd_w = self.kd_weight_at(epoch_frac)
        total = (total + kd_w * kd_loss + self.fa_weight * fa_loss
                 + self.gate_weight * gate_loss)

        if return_individual:
            return (total, picker_loss, polarity_loss, detector_loss,
                    kd_loss.detach(), fa_loss.detach(), gate_loss.detach())
        return total
