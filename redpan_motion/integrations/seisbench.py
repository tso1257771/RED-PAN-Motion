"""
redpan_motion/integrations/seisbench.py

RED-PAN as a SeisBench ``WaveformModel``, so it can be driven through the same
interface as PhaseNet, EQTransformer, GPD and the rest.

Why this is worth having. A quality-control workflow wants agreement between
independently trained pickers as a quality signal, not one picker's own
confidence. SeisBench already gives every model one API over an ObsPy Stream,
so once RED-PAN speaks it, comparing against PhaseNet costs one line instead
of an adapter each time, and SeisBench does the windowing, resampling and
component ordering.

All three checkpoints are wrapped: ``RedpanSB60s`` for the 60 s model and
``RedpanSB90s`` for ``redpan_motion`` and ``edge_rp90``. Both derive from
``_RedpanSBBase``, and on every instance ``variant`` names the checkpoint,
``sp_table`` the threshold table it was fitted with, and ``has_polarity``
whether it outputs first motion.

The polarity head of the 90 s models is the one part of RED-PAN that does not
fit SeisBench cleanly. It reads the vertical component separately from the
picker and detector, demeaned and max-abs scaled but never band-passed, so the
sign of first motion survives. SeisBench hands one normalized tensor to
``annotate_batch_pre``, so the wrapper carries that second copy of Z as a
fourth input channel and splits it again in ``forward``.

Two things do NOT transfer for free, and a wrapper that ignores either scores
worse than the native path for reasons that look like the model:

1. The S-P-adaptive joint thresholds in ``redpan_motion.sp_thresholds``, which
   gate on the MEAN mask over a trigger whose width encodes the S-P time.
   SeisBench's default ``classify_aggregate`` thresholds each phase trace on
   its own. This module overrides it with ``_joint_classify``, so ``classify()``
   returns the pairs ``picks_to_dataframe`` returns on the same probabilities.
   On a set of regional records, the wrapper reproduced nearly all native
   picks within 0.5 s for all three checkpoints, and ``edge_rp90`` also
   emitted a few pairs with no native counterpart. Pass ``sp_adaptive=False`` to keep the same
   pairing with flat thresholds instead of the table.
2. The scaling the polarity stream expects. It is max-abs, not a z-score, and
   leaving it out saturates the head. See ``RedpanSB90s.annotate_batch_pre``.

What still differs from the native path is the long-trace normalization.
``single`` mode normalizes the whole trace with a rolling window before one
forward pass per block, while SeisBench windows first and normalizes per
window, which is what training did. That is the remaining source of the small
pick differences the parity test measures.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

try:
    import seisbench.models as sbm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "seisbench is required for this module: pip install seisbench"
    ) from exc

_CITATION = (
    "Liao, W.-Y., et al. RED-PAN: A Real-Time Earthquake Detection and "
    "Phase-Picking Approach with Multitask Attention Network. "
    "PyTorch port: https://github.com/tso1257771/RED-PAN-Motion"
)

logger = logging.getLogger(__name__)


def _segment_key(trace) -> tuple[str, int, int]:
    """What identifies one annotation segment: the instrument, its start, its length.

    ``annotate`` emits P, S, Detection (and polarity) as channels of one
    prediction array, so the partners of a P segment are the traces that share
    all three. The instrument has to be in the key: every station in a stream
    fetched for one event carries the same window, so start plus length alone
    is the same for all of them. UTCDateTime is not hashable in this ObsPy
    version; integer nanoseconds are, and they are exact where a rounded
    float timestamp is not.
    """
    s = trace.stats
    return (f"{s.network}.{s.station}.{s.location}", int(s.starttime.ns), int(s.npts))


def _as_float64(trace) -> np.ndarray:
    """Trace data as float64 with NaN replaced by 0.0.

    annotate_batch_post writes NaN into the blinded edges. annotate() trims
    the outer ones off, so with the default overlap there are no NaNs left; a
    caller passing overlap below twice the blinding gets holes inside the
    trace. detect_events_joint runs find_peaks and a convolution, so any NaN
    has to go first, and zero is the right filler: it is what "no probability
    here" means to both.
    """
    return np.nan_to_num(np.asarray(trace.data, dtype="float64"), nan=0.0)


def _state_dict_from(ckpt: Any) -> Any:
    """The model state inside a RED-PAN-Motion ``best.pt``.

    The trainer has written three layouts: a bare state dict, or one nested
    under ``model_state_dict`` or ``state_dict``. Anything that is not a dict
    is returned as it is and left to ``load_state_dict`` to reject.
    """
    if isinstance(ckpt, dict):
        return ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    return ckpt


def _joint_classify(model: _RedpanSBBase, annotations, argdict: dict[str, Any]):
    """Native S-P-adaptive pairing over SeisBench annotations.

    ``picks_from_annotations`` thresholds each phase trace on its own, so a P
    with no S, or an S inside no detection, still becomes a pick. The native
    path does not work that way: ``detect_events_joint`` delineates the
    detector mask, and only a detection that contains both a P peak and an S
    peak yields a pair. Which thresholds that detection must clear is chosen
    from its own trigger duration, which is an S-P proxy, so a near event and a
    distant one are not judged on the same numbers.

    Everything here is the pairing, not the probabilities. The annotations are
    whatever ``annotate`` produced, so blinding and window stacking already
    happened and are unchanged.

    ``argdict`` keys, all optional. ``sp_adaptive`` (default True) selects
    the per-bin table; False keeps the same mask-gated pairing but with the
    flat ``P_threshold``, ``S_threshold`` and ``Detection_threshold`` (it is
    NOT SeisBench's per-phase triggering). With the table on, the bin decides
    the thresholds and ``Detection_threshold`` is replaced by the table's
    mask-mean gate; a ``P_threshold`` or ``S_threshold`` the caller passed
    explicitly is still honoured, as a floor on the pair, so the argument
    keeps the meaning a SeisBench user expects. ``min_sp_sec`` (default 1.0)
    drops pairs with S less than that far after P.
    """
    import seisbench.util as sbu

    from ..pairing import detect_events_joint
    from ..sp_thresholds import resolve_table

    def _arg(key: str):
        # The value registered in _annotate_args unless the caller passed the
        # key. The same lookup the stock models use, so each default is
        # written once, in __init__, and this function cannot drift from it.
        return model._argdict_get_with_default(argdict, key)

    table = resolve_table(model.sp_table) if _arg("sp_adaptive") else None
    p_thr, s_thr = _arg("P_threshold"), _arg("S_threshold")
    d_thr = _arg("Detection_threshold")
    min_sp_sec = _arg("min_sp_sec")
    cls = model.__class__.__name__

    def _chan(name):
        return {_segment_key(t): t
                for t in annotations.select(channel=f"{cls}_{name}")}

    p_by, s_by, d_by = _chan("P"), _chan("S"), _chan("Detection")
    pol_by = {n: _chan(f"Polarity_{n}") for n in ("U", "D")} if model.has_polarity else {}

    picks = sbu.PickList()
    for key, tp in p_by.items():
        ts, td = s_by.get(key), d_by.get(key)
        if ts is None or td is None:
            # P, S and Detection are channels of one prediction array with
            # the same blinded edges, so a P segment with no S or Detection
            # partner is a fault in the annotations, not a model decision.
            # It cannot be paired and yields nothing; say so rather than
            # returning an empty result that looks like a quiet record.
            logger.warning("%s: P annotation segment at %s has no matching S or "
                           "Detection segment; it yields no picks",
                           key[0], tp.stats.starttime)
            continue
        p, s, m = _as_float64(tp), _as_float64(ts), _as_float64(td)
        dt = float(tp.stats.delta)
        pairs = detect_events_joint(p, s, m, dt, p_thr, s_thr, d_thr,
                                    sp_thresholds=table)
        min_sp = max(2, round(min_sp_sec / dt))
        pairs = [q for q in pairs if q[2] - q[0] >= min_sp]
        if table is not None:
            # detect_events_joint lowers p_thr / s_thr to the table's floor
            # and lets the bin decide, so the values above never gate a pair
            # in this mode. classify_async hands classify_aggregate only the
            # keys the caller passed (defaults are not merged in), so a key
            # present here is an explicit request and is applied as a floor.
            fp, fs = argdict.get("P_threshold"), argdict.get("S_threshold")
            pairs = [q for q in pairs
                     if (fp is None or q[1] >= fp) and (fs is None or q[3] >= fs)]

        trace_id = key[0]
        t0 = tp.stats.starttime
        pol_u = pol_by.get("U", {}).get(key)
        pol_d = pol_by.get("D", {}).get(key)
        if pol_u is not None and pol_d is not None:
            # Same key, so the same length as p: an index that came out of
            # find_peaks on p is in range here too.
            pol_u, pol_d = _as_float64(pol_u), _as_float64(pol_d)
        else:
            pol_u = pol_d = None

        for p_idx, p_pv, s_idx, s_pv in pairs:
            for phase, idx, pv, prob in (("P", p_idx, p_pv, p), ("S", s_idx, s_pv, s)):
                lo, hi = _run_around(prob, idx, pv / 2.0)
                pk = {"trace_id": trace_id,
                      "start_time": t0 + lo * dt, "end_time": t0 + hi * dt,
                      "peak_time": t0 + idx * dt, "peak_value": pv, "phase": phase}
                if phase == "P" and pol_u is not None and pol_d is not None:
                    # "up"/"down" is SeisBench's own vocabulary, the keys of
                    # _PYROCKO_POLARITY_MAP. U against D, ignoring the head's
                    # N channel, is what picks_to_dataframe compares.
                    up, dn = float(pol_u[idx]), float(pol_d[idx])
                    pk["polarity"] = "up" if up >= dn else "down"
                    pk["polarity_value"] = max(up, dn)
                picks.append(sbu.Pick(**pk))
    return sbu.ClassifyOutput(model.name, picks=sbu.PickList(sorted(picks)))


def _run_around(prob: np.ndarray, idx: int, level: float) -> tuple[int, int]:
    """The contiguous span around ``idx`` where ``prob`` stays above ``level``.

    Gives the pick a start and an end, which SeisBench's own picks take from a
    trigger. Half the peak value is the same convention
    ``picks_from_annotations`` uses for its lower trigger.
    """
    n = len(prob)
    lo = idx
    while lo > 0 and prob[lo - 1] >= level:
        lo -= 1
    hi = idx
    while hi < n - 1 and prob[hi + 1] >= level:
        hi += 1
    return lo, hi


def _highpass_batch(batch: torch.Tensor, freq: float, dt: float) -> torch.Tensor:
    """Demean then 4th-order zero-phase Butterworth highpass, over the last axis.

    The vectorized equivalent of ``redpan_motion.signal.highpass`` applied to
    each channel of each window. scipy has no torch backend, so the batch makes
    a round trip through numpy. That is the dominant cost of
    ``annotate_batch_pre`` and is why the filter is not applied per sample.
    """
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, freq / (0.5 / dt), btype="high", output="sos")
    arr = batch.detach().cpu().numpy().astype("float64")
    arr = arr - arr.mean(axis=-1, keepdims=True)
    arr = sosfiltfilt(sos, arr, axis=-1)
    return torch.from_numpy(arr.astype("float32")).to(batch.device)


class _RedpanSBBase(sbm.WaveformModel):
    """What the RED-PAN wrappers share.

    The annotate arguments, blinding, the classify pairing, the per-window
    normalization, SeisBench save and load, and checkpoint loading are the same
    for every checkpoint and live here once. A subclass states its checkpoints
    in ``VARIANTS`` and ``SP_TABLES``, its output ``LABELS`` and default window,
    and how its backbone is built, fed and overlapped.

    ``variant`` names the checkpoint an instance wraps. ``sp_table`` and
    ``has_polarity`` follow from it and from ``labels``, so they read the same
    way on every subclass and on every instance.
    """

    _weight_warnings: ClassVar[list] = []

    #: checkpoint directory name -> builder function in redpan_motion.models
    VARIANTS: ClassVar[dict[str, str]] = {}
    #: checkpoint directory name -> table in redpan_motion.sp_thresholds
    SP_TABLES: ClassVar[dict[str, str]] = {}
    #: output channels, in the order forward() returns them
    LABELS: ClassVar[tuple[str, ...]] = ()
    #: window length in samples when a checkpoint's config does not state one
    DEFAULT_IN_SAMPLES: ClassVar[int] = 0

    def __init__(
        self,
        *,
        variant: str,
        in_samples: int,
        sampling_rate: float,
        highpass_freq: float,
        model_kwargs: dict | None,
        **kwargs,
    ):
        if variant not in self.VARIANTS:
            raise ValueError(
                f"variant must be one of {sorted(self.VARIANTS)}, got {variant!r}")
        super().__init__(
            citation=_CITATION,
            output_type="array",
            in_samples=in_samples,
            pred_sample=(0, in_samples),
            labels=list(self.LABELS),
            sampling_rate=sampling_rate,
            component_order="ENZ",
            grouping="instrument",
            **kwargs,
        )
        self.variant = variant
        self.in_samples = in_samples
        self.highpass_freq = highpass_freq
        # Kept on the instance because get_model_args must return it: SeisBench
        # load() rebuilds through cls(**model_args), and a checkpoint whose
        # config differs from the builder defaults (redpan_motion's filter
        # count does) cannot be loaded into the default architecture.
        self.model_kwargs = dict(model_kwargs or {})

        # _annotate_args is a CLASS attribute on WaveformModel, so mutating
        # self._annotate_args in place would edit the dict every other model
        # inherits, including WaveformModel's own. The stock models avoid this
        # by copying at class level (see PhaseNet); these values depend on
        # in_samples, so the copy is per instance. Without it, building a 90 s
        # model silently rewrote a 60 s model's overlap and blinding, and the
        # 60 s model then stopped reporting P.
        self._annotate_args = dict(self._annotate_args)

        # Registered through _annotate_args rather than default_args, the way
        # the stock models do it. The two are NOT the same mechanism:
        # default_args does not populate _annotate_args, so a key placed only
        # there is absent from every _annotate_args lookup. That is easy to get
        # wrong and hard to see, because an exception raised inside annotate's
        # async worker is swallowed and the pipeline deadlocks instead.
        #
        # Blinding discards the edges of each window, where a U-Net has least
        # context. Overlap is the subclass's choice; see _overlap.
        blind = in_samples // 12
        self._annotate_args["overlap"] = (
            self._annotate_args["overlap"][0], self._overlap(in_samples))
        self._annotate_args["blinding"] = (
            ("Number of prediction samples to discard on each side of each "
             "window prediction"), (blind, blind))
        self._annotate_args["*_threshold"] = (
            "Detection threshold for the provided phase", 0.3)
        self._annotate_args["P_threshold"] = ("Detection threshold for P", 0.3)
        self._annotate_args["S_threshold"] = ("Detection threshold for S", 0.2)
        self._annotate_args["Detection_threshold"] = (
            "Threshold on the event mask", 0.5)
        # classify-only keys. Registered so annotate's argument check does not
        # log "Unknown argument ... will be ignored" for them, which is false:
        # classify_aggregate reads both.
        self._annotate_args["sp_adaptive"] = (
            ("Use the S-P-adaptive joint threshold table (classify only); False "
             "keeps the mask-gated pairing with the flat thresholds"), True)
        self._annotate_args["min_sp_sec"] = (
            ("Drop a pair whose S is less than this many seconds after P "
             "(classify only)"), 1.0)

        self.model = self._build_model()

    # -- what a subclass provides -----------------------------------------

    def _overlap(self, in_samples: int) -> int:
        """Samples of overlap between consecutive annotate windows."""
        raise NotImplementedError

    def _build_model(self) -> torch.nn.Module:
        """The backbone, built from self.variant and self.model_kwargs."""
        raise NotImplementedError

    @classmethod
    def _complete_model_kwargs(
        cls, model_kwargs: dict, state: dict, accepted: set[str]
    ) -> None:
        """Fill builder arguments a checkpoint's config leaves out. In place."""

    # -- identity -----------------------------------------------------------

    @property
    def sp_table(self) -> str:
        """The table in redpan_motion.sp_thresholds this checkpoint was fitted with."""
        return self.SP_TABLES[self.variant]

    @property
    def has_polarity(self) -> bool:
        """Whether this model outputs first motion, read off its labels."""
        return any(label.startswith("Polarity_") for label in self.labels)

    # -- SeisBench contract -------------------------------------------------

    @staticmethod
    def _zscore(x: torch.Tensor) -> torch.Tensor:
        """Per-channel z-score over each window, as the native path does it.

        Population standard deviation and a floor that replaces a std below
        1e-8 with 1.0, both matching ``REDPANPredictor._prepare_input`` and the
        training loader. torch's own default is the sample estimate, and the
        earlier ``x / (std + 1e-10)`` scaled a near-dead channel up to unit
        variance where the native path leaves it near zero.
        """
        x = x - x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, correction=0)
        return x / torch.where(std < 1e-8, torch.ones_like(std), std)

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = torch.transpose(batch, -1, -2)    # (B, T, labels)
        prenan, postnan = argdict.get("blinding", self._annotate_args["blinding"][1])
        if prenan > 0:
            batch[:, :prenan] = torch.nan
        if postnan > 0:
            batch[:, -postnan:] = torch.nan
        return batch

    def classify_aggregate(self, annotations, argdict):
        """The native mask-gated, S-P-adaptive pairing. See _joint_classify.

        classify() returns the pairs picks_to_dataframe returns on the same
        probabilities. When the model has a polarity head, each P pick also
        carries first motion as Pick.polarity and Pick.polarity_value. Pass
        sp_adaptive=False for the same pairing with flat thresholds. With the
        table on, P_threshold and S_threshold apply only when passed
        explicitly, as floors.
        """
        return _joint_classify(self, annotations, argdict)

    def get_model_args(self):
        """Constructor arguments, for SeisBench save() and load().

        load() rebuilds through cls(**model_args), so everything that changes
        behaviour or architecture has to come back, model_kwargs included.
        """
        args = super().get_model_args()
        for k in ("citation", "output_type", "pred_sample", "labels",
                  "component_order", "grouping", "default_args"):
            args.pop(k, None)
        args.update(variant=self.variant, in_samples=self.in_samples,
                    sampling_rate=self.sampling_rate,
                    highpass_freq=self.highpass_freq,
                    model_kwargs=self.model_kwargs)
        return args

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_redpan_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device: str | None = None,
        variant: str | None = None,
    ) -> _RedpanSBBase:
        """Build from a RED-PAN-Motion ``checkpoints/<variant>/`` directory.

        The architecture is rebuilt from the sibling ``config.json``,
        forwarding only the keys the builder accepts, the same rule
        ``REDPANPredictor.from_checkpoint`` follows, so the two paths cannot
        disagree about it.

        ``variant`` defaults to the directory name. A class that wraps a
        single checkpoint accepts any directory name, as the 60 s wrapper
        always has; one that wraps several needs the name, or ``variant=``.
        """
        from .. import models as rpm

        d = Path(checkpoint_dir)
        if variant is None:
            if d.name in cls.VARIANTS:
                variant = d.name
            elif len(cls.VARIANTS) == 1:
                variant = next(iter(cls.VARIANTS))
            else:
                raise ValueError(
                    f"{d} is not named after a checkpoint {cls.__name__} wraps; "
                    f"pass variant= one of {sorted(cls.VARIANTS)}")
        elif variant not in cls.VARIANTS:
            raise ValueError(
                f"variant must be one of {sorted(cls.VARIANTS)}, got {variant!r}")

        cfg = {}
        if (d / "config.json").exists():
            cfg = json.loads((d / "config.json").read_text())
        input_size = cfg.get("input_size") or [cls.DEFAULT_IN_SAMPLES, 3]

        builder = getattr(rpm, cls.VARIANTS[variant])
        accepted = set(inspect.signature(builder).parameters)
        model_kwargs = {k: v for k, v in cfg.items() if k in accepted}

        # weights_only: every shipped checkpoint is tensors, numbers and plain
        # dicts, so nothing in it needs arbitrary unpickling to load.
        ckpt = torch.load(d / "best.pt", map_location="cpu", weights_only=True)
        state = _state_dict_from(ckpt)
        cls._complete_model_kwargs(model_kwargs, state, accepted)

        obj = cls(variant=variant, in_samples=int(input_size[0]),
                  model_kwargs=model_kwargs)
        obj.model.load_state_dict(state)
        obj.eval()
        if device:
            obj.to(device)
        return obj


class RedpanSB60s(_RedpanSBBase):
    """The original RED-PAN 60 s model behind the SeisBench interface.

    Outputs four per-sample traces in ``labels`` order: P, S, N and Detection.
    The underlying model returns the picker as (B, 3, T) softmax over P/S/N
    and the detector as (B, 2, T) softmax over mask and no-mask. Only the mask
    channel of the detector is exposed, because its complement carries no
    extra information and an extra label would appear as a channel that
    ``classify_aggregate`` has to be told to ignore.

    Component order is ENZ, matching ``redpan_motion.waveform_io.ENZ_ORDER``
    and ``CH_Z = 2``.
    """

    VARIANTS: ClassVar[dict[str, str]] = {"redpan_60s": "build_redpan_60s"}
    SP_TABLES: ClassVar[dict[str, str]] = {"redpan_60s": "redpan_60s"}
    LABELS: ClassVar[tuple[str, ...]] = ("P", "S", "N", "Detection")
    DEFAULT_IN_SAMPLES: ClassVar[int] = 6000

    def __init__(
        self,
        in_samples: int = 6000,
        sampling_rate: float = 100.0,
        highpass_freq: float = 1.0,
        model_kwargs: dict | None = None,
        variant: str = "redpan_60s",
        **kwargs,
    ):
        super().__init__(variant=variant, in_samples=in_samples,
                         sampling_rate=sampling_rate, highpass_freq=highpass_freq,
                         model_kwargs=model_kwargs, **kwargs)

    def _overlap(self, in_samples: int) -> int:
        # Half the window, the usual choice for an array model of this
        # length. The parity measured for this checkpoint was measured with
        # it, so it stays, although the 90 s class uses the minimum instead.
        return in_samples // 2

    def _build_model(self) -> torch.nn.Module:
        from ..models import build_redpan_60s
        # in_samples is what SeisBench windows on, so it decides the input
        # size; model_kwargs carries the rest of a checkpoint's config.
        return build_redpan_60s(
            **{**self.model_kwargs, "input_size": (self.in_samples, 3)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, T) waveform -> (B, 4, T) in labels order."""
        picker, detector = self.model(x)          # (B,3,T) P/S/N ; (B,2,T) mask
        return torch.cat([picker, detector[:, :1]], dim=1)

    def annotate_batch_pre(self, batch: torch.Tensor, argdict: dict[str, Any]) -> torch.Tensor:
        """1 Hz highpass, then the per-window z-score. See _zscore.

        The highpass matches the preprocessing the native path uses, and the
        z-score is what the models were trained on, over each training window.
        """
        if self.highpass_freq:
            batch = _highpass_batch(batch, self.highpass_freq, 1.0 / self.sampling_rate)
        return self._zscore(batch)


class RedpanSB90s(_RedpanSBBase):
    """The 90 s models (``redpan_motion`` and ``edge_rp90``) behind SeisBench.

    Seven per-sample traces: P, S, N, Detection, and the polarity softmax over
    N, U and D at every sample. Read polarity at the P index, which is what
    ``redpan_motion.picks`` does, and what ``classify()`` puts on each P pick.

    HOW RAW Z SURVIVES THE PIPELINE. The polarity head is trained on the
    unfiltered vertical, because filtering softens the sign of first motion.
    SeisBench normalizes everything in ``annotate_batch_pre`` before the model
    sees it, which is exactly what must not happen to Z.

    The way through is that SeisBench passes whatever ``annotate_batch_pre``
    returns straight to ``forward`` (``predictions = self(preprocessed)``),
    without checking the channel count. So ``annotate_batch_pre`` returns FOUR
    channels: the three processed components the backbone needs, plus a copy
    of Z that is only demeaned and max-abs scaled. ``forward`` splits them
    again and hands the copy to the model as ``z_raw``.

    This keeps the raw path stateless. Stashing Z on ``self`` between pre and
    forward would also work on paper, but ``annotate`` runs its batches through
    an asyncio pipeline, so anything carried on the instance is a race waiting
    to happen. A channel travels with its own batch and cannot be mismatched.

    ONE DIFFERENCE FROM THE NATIVE PATH. The native deploy filters the whole
    trace and then windows it. Here the window comes first, so the highpass is
    applied per window and each window carries its own filter transient at the
    edges. Blinding discards ``in_samples // 12``, which is 7.5 s at each end,
    and the transient of a 1 Hz highpass is far shorter than that, so the
    discarded region absorbs it. The parity test is what confirms it rather
    than assuming it.
    """

    VARIANTS: ClassVar[dict[str, str]] = {
        "redpan_motion": "build_mtan_r2unet_rp90_motion",
        "edge_rp90": "build_edge_rp90",
    }
    SP_TABLES: ClassVar[dict[str, str]] = {
        "redpan_motion": "rp90_motion_v49",
        "edge_rp90": "edge_rp90",
    }
    LABELS: ClassVar[tuple[str, ...]] = (
        "P", "S", "N", "Detection", "Polarity_N", "Polarity_U", "Polarity_D")
    DEFAULT_IN_SAMPLES: ClassVar[int] = 9000

    def __init__(
        self,
        variant: str = "redpan_motion",
        in_samples: int = 9000,
        sampling_rate: float = 100.0,
        highpass_freq: float = 1.0,
        model_kwargs: dict | None = None,
        **kwargs,
    ):
        super().__init__(variant=variant, in_samples=in_samples,
                         sampling_rate=sampling_rate, highpass_freq=highpass_freq,
                         model_kwargs=model_kwargs, **kwargs)

    def _overlap(self, in_samples: int) -> int:
        # Overlap must be at least twice the blinding, or the output has holes.
        # Blinding discards in_samples // 12 samples at each window edge. With
        # no overlap the windows tile edge to edge, every boundary loses twice
        # that, and the annotation comes back with a periodic gap (15 s every
        # 90 s). Twice the blinding makes the unblinded parts exactly
        # contiguous.
        #
        # It is kept at that minimum rather than the usual half window. On a
        # set of regional records, a half-window overlap lost several native
        # picks that the minimum overlap reproduced: the native path makes a
        # single pass, and averaging many
        # overlapping windows moves the joint mask, P and S decision enough to
        # lose pairs.
        return 2 * (in_samples // 12)

    def _build_model(self) -> torch.nn.Module:
        from .. import models as rpm
        builder = getattr(rpm, self.VARIANTS[self.variant])
        return builder(**self.model_kwargs)

    @classmethod
    def _complete_model_kwargs(
        cls, model_kwargs: dict, state: dict, accepted: set[str]
    ) -> None:
        # The polarity head width is not always in the config; recover it from
        # the checkpoint, as REDPANPredictor.from_checkpoint does.
        for key in ("polarity_head.weight", "polarity.head.weight"):
            if key in state and "polarity_output_channels" in accepted:
                model_kwargs.setdefault("polarity_output_channels",
                                        int(state[key].shape[0]))
        if "use_polarity" in accepted:
            model_kwargs.setdefault("use_polarity", True)

    def annotate_batch_pre(self, batch: torch.Tensor, argdict: dict[str, Any]) -> torch.Tensor:
        """(B, 3, T) raw counts -> (B, 4, T), the 4th channel a demeaned,
        max-abs scaled Z.

        The first three channels get the 1 Hz highpass and the per-window
        z-score, as the 60 s model does. The 4th channel is NOT filtered,
        because filtering softens the sign of first motion. It is demeaned and
        max-abs scaled, which is what ``REDPANPredictor._z_raw_from`` does to
        the raw vertical.

        Demeaning is required, not optional. Raw counts carry a DC offset of
        thousands, and the polarity head splits its input on sign, so an
        offset that large decides every sample before the waveform is even
        considered. Passing true raw counts here gives a polarity stream that
        correlates with the native one at about zero.

        Scaling is required too, and by max absolute value rather than by
        standard deviation. That is the training convention: divide by the
        largest absolute sample so the stream lands in [-1, 1] with its sign
        intact. Feeding raw counts of order 1e5 saturates the head: on a
        record where the native path called the first motion up with high
        confidence, the unscaled wrapper returned exactly [1, 0, 0] at the
        same sample from the same window, with an identical P probability.

        The native path takes the maximum over the whole trace, this one over
        each window, because SeisBench has no whole trace to look at. On a
        record whose largest vertical swing is the event itself the two agree.
        On a set of regional to teleseismic records, the window holding the
        P often had a smaller maximum than the whole trace. The three-way
        N/U/D distribution then shifted noticeably, but the up/down call
        that classify() reports did not change on any of them.
        """
        z_raw = batch[:, 2:3]
        z_raw = z_raw - z_raw.mean(dim=-1, keepdim=True)
        zmax = z_raw.abs().amax(dim=-1, keepdim=True)
        z_raw = z_raw / torch.where(zmax < 1e-9, torch.ones_like(zmax), zmax)
        x = batch
        if self.highpass_freq:
            x = _highpass_batch(batch, self.highpass_freq, 1.0 / self.sampling_rate)
        return torch.cat([self._zscore(x), z_raw], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, T) -> (B, 7, T) in labels order."""
        waveform, z_raw = x[:, :3], x[:, 3:4]
        picker, polarity, detector = self.model(waveform, z_raw=z_raw)
        return torch.cat([picker, detector[:, :1], polarity], dim=1)
