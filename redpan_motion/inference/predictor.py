"""Inference for PyTorch RED-PAN: predictions on seismic waveforms."""

import logging
import torch
import torch.nn as nn
import numpy as np
import pickle
import warnings
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)
from obspy import Stream, Trace, UTCDateTime

from redpan_motion.models.mtan_r2unet import MTAN_R2UNet
from redpan_motion.models.mtan_r2unet_rp90_motion import (
    build_mtan_r2unet_rp90_motion,
)
from redpan_motion.utils.waveform import (
    pad_waveform_with_noise,
    sac_len_complement,
    create_gaussian_weights,
    create_triangular_weights,
    create_cosine_weights,
)
from redpan_motion.checkpoints import resolve as _resolve_checkpoint
from redpan_motion.sp_thresholds import (
    resolve_table as _sp_resolve_table,
    params_for_duration as _sp_params_for_duration,
)


# Rolling-window default, per checkpoint.
#
# The rolling window divides every sample by a centred moving std, which
# lifts pre-event noise to unit amplitude so that noise and signal regions
# are on one scale. On a low-SNR distant record that also flattens the
# onset, and EdgeRP90, at 309k parameters, then loses the P outright where
# the larger checkpoints only lose contrast.
#
# On a set of regional to teleseismic records, a window longer than
# pred_npts let EdgeRP90 find the P on several distant records it missed at
# pred_npts, and each gained peak lay close to the predicted P arrival. A
# 30000-sample window was no worse than the longer windows tried. The other
# two checkpoints changed little across the window lengths, and Redpan60s did
# marginally better at its own pred_npts, so this is set per checkpoint rather
# than globally.
_ROLLING_WINDOW_BY_MODEL = {"EdgeRP90": 30000}


class REDPANPredictor:
    """
    PyTorch RED-PAN predictor for seismic phase picking.
    
    Provides the same interface as the REDPAN class of the original TensorFlow RED-PAN.
    
    Example:
        predictor = REDPANPredictor.from_checkpoint('redpan_motion')   # or a path to a .pt
        picker, detector = predictor.predict(stream)
    """
    
    def __init__(
        self,
        model: nn.Module,
        pred_npts: int = 6000,
        dt: float = 0.01,
        pred_interval_sec: float = 10.0,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        output_order: str = 'NPS',
        picker_output_permutation: Optional[Tuple[int, int, int]] = None,
        detector_output_permutation: Optional[Tuple[int, int]] = None,
        # Single-entry refine parameters
        refine_trigger_on: float = 0.3,
        refine_trigger_off: float = 0.3,
        refine_pre_trigger_sec: float = 5.0,
        refine_smooth_npts: int = 10,
        # Continuous-inference strategy
        inference_mode: str = "sliding",
        window_weight: str = "gaussian",
        single_pass_block_sec: float = 600.0,
        # Threshold post-processing (applied when postprocess=True)
        mask_threshold: float = 0.5,
        p_threshold: float = 0.3,
        s_threshold: float = 0.3,
        detect_threshold: float = 0.1,
        sp_adaptive_thresholds: Union[None, bool, str, dict] = None,
    ):
        """
        Args:
            model: The MTAN_R2UNet model
            pred_npts: Model input length (samples)
            dt: Sample interval (seconds)
            pred_interval_sec: Sliding window step for long waveforms
            batch_size: Batch size for prediction
            device: Device for inference
            output_order: Channel order for picker output ('NPS' or 'PSN')
            picker_output_permutation: Optional explicit channel permutation for picker
            detector_output_permutation: Optional explicit channel permutation for detector
            refine_trigger_on: Mask probability threshold to START a trigger for
                single-entry refine (default 0.3). Lower to refine more events;
                raise to restrict refine to high-confidence detections only.
            refine_trigger_off: Mask probability threshold to END a trigger
                (default 0.3). Usually set equal to refine_trigger_on.
            refine_pre_trigger_sec: How many seconds before trigger onset the
                focused refine window starts (default 5.0 s). Increase if P
                arrivals tend to occur well before the mask trigger onset.
            refine_smooth_npts: Moving-average kernel length (samples) applied
                to the mask before trigger detection (default 10). Increase to
                suppress noisy mask fluctuations that create spurious triggers.
        """
        self._last_polarity = None    # set by predict_array when model has polarity
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()

        # z_raw (sign-preserved, max-abs-scaled Z) feeds the polarity stream so
        # first-motion matches the training convention. Detect support + Z index.
        _base = self.model
        for _a in ('module', '_orig_mod'):
            if hasattr(_base, _a):
                _base = getattr(_base, _a)
        self._ch_z = int(getattr(_base, 'CH_Z', 2))
        try:
            import inspect as _inspect
            self._accepts_z_raw = 'z_raw' in _inspect.signature(_base.forward).parameters
        except (ValueError, TypeError):
            self._accepts_z_raw = False

        if pred_npts <= 0:
            raise ValueError(f"pred_npts must be positive; got {pred_npts}")
        self.pred_npts = pred_npts
        self.dt = dt
        self.pred_interval_sec = pred_interval_sec
        self.pred_interval_npts = int(round(pred_interval_sec / dt))
        self.batch_size = batch_size
        self.output_order = output_order.upper()
        if self.output_order not in {'NPS', 'PSN'}:
            raise ValueError("output_order must be 'NPS' or 'PSN'")
        self.picker_output_permutation = picker_output_permutation
        self.detector_output_permutation = detector_output_permutation

        self.refine_trigger_on      = refine_trigger_on
        self.refine_trigger_off     = refine_trigger_off
        self.refine_pre_trigger_sec = refine_pre_trigger_sec
        self.refine_smooth_npts     = refine_smooth_npts

        # Continuous-inference strategy: "sliding" (overlapping windows with
        # tapered overlap-add — the original RED-PAN style, smoother output) or
        # "single" (one forward pass per large block — leverages the model's
        # built-in length flexibility; far fewer FLOPs, no overlap blending).
        if inference_mode not in {"sliding", "single"}:
            raise ValueError("inference_mode must be 'sliding' or 'single'")
        self.inference_mode = inference_mode
        self.window_weight = window_weight
        self.position_weights = self._make_position_weights(window_weight)
        self.single_pass_block = int(round(single_pass_block_sec / dt))
        # Long-trace ("single" mode) normalization: "rolling" (default — each sample
        # normalized by its centered pred_npts window via a moving mean/std; a big
        # earthquake then only inflates its own ~window neighbourhood, not a whole 600s
        # block), "global" (legacy per-block z-score), or "moving" (PhaseNet+ L1 moving
        # normalize, filter=moving_filter_size ~10s — for models trained normalize_mode="moving").
        # Rolling reproduces the per-90s training-window z-score continuously (ported
        # from redpan/torch rolling_normalize); moving matches EQNet/dataset_v2 moving_normalize.
        self.long_trace_norm = "rolling"
        # Keyed on the unwrapped class (``_base``), so a DataParallel or
        # torch.compile wrapper does not silently fall back to pred_npts.
        self.rolling_window = _ROLLING_WINDOW_BY_MODEL.get(
            type(_base).__name__, self.pred_npts)
        self.moving_filter_size = 1024

        # Threshold post-processing defaults.
        self.mask_threshold = mask_threshold
        self.p_threshold = p_threshold
        self.s_threshold = s_threshold
        self.detect_threshold = detect_threshold
        # S-P-adaptive joint thresholds: True auto-selects the table from the model class;
        # a str/dict is resolved explicitly; None keeps the fixed thresholds above.
        if sp_adaptive_thresholds is True:
            sp_adaptive_thresholds = type(_base).__name__
        self.sp_thresholds = _sp_resolve_table(sp_adaptive_thresholds)
        self._last_events = None

    def _make_position_weights(self, kind: str) -> np.ndarray:
        """Overlap-add position weights for sliding-window accumulation.

        'gaussian'/'cosine'/'triangular' taper toward the window edges (the
        original RED-PAN style — down-weights unreliable boundary predictions for
        smoother seams); 'uniform' is the flat SeisBench-style accumulation.
        """
        kind = (kind or "gaussian").lower()
        if kind == "uniform":
            return np.ones(self.pred_npts, dtype=np.float32)
        if kind == "gaussian":
            return create_gaussian_weights(self.pred_npts)
        if kind == "cosine":
            return create_cosine_weights(self.pred_npts)
        if kind == "triangular":
            return create_triangular_weights(self.pred_npts)
        raise ValueError(
            f"window_weight must be gaussian/cosine/triangular/uniform; got {kind!r}")

    @classmethod
    def _load_checkpoint_compat(cls, checkpoint_path: str, map_location: str = 'cpu'):
        """Load checkpoint across PyTorch versions (2.6 defaults to weights_only=True)."""
        try:
            return torch.load(
                checkpoint_path, map_location=map_location, weights_only=True)
        except (pickle.UnpicklingError, RuntimeError) as exc:
            msg = str(exc)
            if "Weights only load failed" not in msg:
                raise

            # First try keep-safe loading by allowlisting our training config class.
            try:
                from redpan_motion.training.trainer import TrainingConfig
                safe_globals = getattr(torch.serialization, "safe_globals", None)
                if safe_globals is not None:
                    with safe_globals([TrainingConfig]):
                        return torch.load(
                            checkpoint_path, map_location=map_location,
                            weights_only=True)
            except Exception as exc:
                logger.debug(
                    "safe_globals load failed for %s (%s); falling back to "
                    "weights_only=False", checkpoint_path, exc)

            # Fallback for trusted local checkpoints created by RED-PAN trainer.
            warnings.warn(
                "Falling back to torch.load(weights_only=False) for checkpoint compatibility. "
                "Only do this for trusted checkpoints.",
                RuntimeWarning,
            )
            try:
                return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            except TypeError:
                # Older torch versions may not have weights_only argument.
                return torch.load(checkpoint_path, map_location=map_location)

    @staticmethod
    def _load_sibling_config(checkpoint_path: str) -> dict:
        """Try to load config.json from the checkpoint's directory or parent."""
        import json
        from pathlib import Path
        ckpt = Path(checkpoint_path)
        for candidate in (ckpt.parent / 'config.json', ckpt.parent.parent / 'config.json'):
            if candidate.is_file():
                try:
                    with open(candidate) as f:
                        return json.load(f)
                except Exception as exc:
                    logger.warning(
                        "failed to read config %s (%s)", candidate, exc)
        return {}

    @staticmethod
    def _strip_state_dict_prefixes(sd: dict) -> dict:
        """Strip wrapper prefixes from DDP/torch.compile saved checkpoints."""
        prefixes = ('_orig_mod.module.', 'module._orig_mod.', '_orig_mod.', 'module.')
        sample_key = next(iter(sd), '')
        for prefix in prefixes:
            if sample_key.startswith(prefix):
                return {k[len(prefix):]: v for k, v in sd.items()}
        return sd

    def _z_raw_from(self, raw_z: np.ndarray) -> torch.Tensor:
        """Build the polarity stream's z_raw input: (B, T) raw vertical component
        -> (B, 1, T) sign-preserved, max-abs scaled to [-1, 1] (training convention)."""
        z = np.asarray(raw_z, dtype=np.float32)
        if z.ndim == 1:
            z = z[None, :]
        zmax = np.max(np.abs(z), axis=1, keepdims=True)
        zmax[zmax < 1e-9] = 1.0
        return torch.from_numpy((z / zmax)[:, None, :]).to(self.device)

    def _model_forward(self, x: torch.Tensor, z_raw: Optional[torch.Tensor] = None):
        """Unified model forward: handles 2/3/4-output models.
        Always returns (picker, detector, polarity_or_None)."""
        if z_raw is not None and self._accepts_z_raw:
            out = self.model(x, z_raw=z_raw)
        else:
            out = self.model(x)
        if isinstance(out, tuple) and len(out) == 3:
            picker, polarity, detector = out
            return picker, detector, polarity
        elif isinstance(out, tuple) and len(out) >= 4:
            # legacy return_ungated_polarity=True: (picker, polarity, detector, raw_pol)
            picker, polarity, detector = out[0], out[1], out[2]
            return picker, detector, polarity
        else:
            picker, detector = out
            return picker, detector, None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_kwargs: Optional[dict] = None,
        model_type: Optional[str] = None,
        **kwargs,
    ) -> 'REDPANPredictor':
        """
        Load predictor from a checkpoint file.

        The architecture is rebuilt from the sibling ``config.json`` (or the
        checkpoint-embedded config / explicit model_kwargs). All four shipped
        families load through here: EdgeRP90 (``edge_rp90_v1``), the 60 s port
        (``redpan_60s``), RP90-Motion (``mtan_r2unet_rp90_motion``, the one
        with the polarity stream), and the base ``mtan_r2unet``.

        Args:
            checkpoint_path: Path to a .pt checkpoint, or the name of a shipped
                one: 'redpan_60s', 'redpan_motion' or 'edge_rp90'.
            model_kwargs: Optional model init arguments (override config values).
            model_type: Explicit model type override ('edge_rp90_v1',
                'redpan_60s', 'mtan_r2unet_rp90_motion' or 'mtan_r2unet').
                If None, auto-detects from state_dict keys.
            **kwargs: Additional arguments for REDPANPredictor

        Returns:
            Initialized predictor
        """
        model_kwargs = model_kwargs or {}
        # A shipped checkpoint may be named rather than located.
        checkpoint_path = str(_resolve_checkpoint(checkpoint_path, file=True))

        checkpoint = cls._load_checkpoint_compat(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and any(
            k.startswith(('init_rrconv', 'bottleneck', 'enc_')) for k in checkpoint
        ):
            state_dict = checkpoint
        else:
            state_dict = checkpoint

        state_dict = cls._strip_state_dict_prefixes(state_dict)

        # Collect config from multiple sources (lowest to highest priority):
        # 1. Sibling config.json next to checkpoint
        # 2. Checkpoint-embedded config object
        # 3. Explicit model_kwargs from caller
        file_config = cls._load_sibling_config(checkpoint_path)

        ckpt_config = {}
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            cfg_obj = checkpoint['config']
            if isinstance(cfg_obj, dict):
                ckpt_config = cfg_obj
            elif hasattr(cfg_obj, '__dict__'):
                ckpt_config = vars(cfg_obj)

        # Resolve model_type: explicit > checkpoint config > file config > auto-detect
        resolved_type = model_type
        if resolved_type is None:
            resolved_type = ckpt_config.get('model_type') or file_config.get('model_type')
        if resolved_type is None:
            # Auto-detect from state_dict keys. Each family owns a prefix the
            # others never use: EdgeRP90 has stem_blocks/neck; the 60 s port is
            # a conversion of the TF graph, so its whole key namespace
            # (ps_init/m_init/pred_ps/...) is disjoint from the MTAN line's;
            # RP90-Motion has pol_init_rrconv; else the base MTAN_R2UNet.
            if any(k.startswith('stem_blocks.') or k.startswith('neck.')
                   for k in state_dict):
                resolved_type = 'edge_rp90_v1'
            elif any(k.startswith('ps_init.') or k.startswith('pred_ps.')
                     for k in state_dict):
                resolved_type = 'redpan_60s'
            else:
                has_pol_rrconv = any(k.startswith('pol_init_rrconv.') for k in state_dict)
                resolved_type = ('mtan_r2unet_rp90_motion' if has_pol_rrconv
                                 else 'mtan_r2unet')

        if resolved_type == 'edge_rp90_v1':
            from redpan_motion.models import build_edge_rp90
            edge_kwargs = {}
            for key in ('input_size', 'block', 'polarity_head',
                        'polarity_output_channels', 'dropout_rate', 'pad_mode'):
                for src in (model_kwargs, ckpt_config, file_config):
                    if key in src:
                        edge_kwargs[key] = src[key]
                        break
            if ('polarity_output_channels' not in edge_kwargs
                    and 'polarity.head.weight' in state_dict):
                edge_kwargs['polarity_output_channels'] = int(
                    state_dict['polarity.head.weight'].shape[0])
            model = build_edge_rp90(**edge_kwargs)
        elif resolved_type == 'redpan_60s':
            from redpan_motion.models import build_redpan_60s
            # Redpan60s is a fixed conversion of the released TF graph: every
            # other architecture key in its config (nb_filters, kernel_size,
            # stride_size, upsize, rrconv_time, ...) describes that graph and is
            # already baked into the class, so input_size is the only one worth
            # forwarding. build_redpan_60s swallows the rest via **_ignored.
            sixty_kwargs = {}
            for src in (model_kwargs, ckpt_config, file_config):
                if 'input_size' in src:
                    sixty_kwargs['input_size'] = src['input_size']
                    break
            model = build_redpan_60s(**sixty_kwargs)
        elif resolved_type in ('mtan_r2unet_rp90_motion', 'mtan_r2unet_rp90_motion_xl'):
            has_pol = any(k.startswith('pol_init_rrconv.') or k.startswith('polarity_head.')
                          for k in state_dict)
            rp90_kwargs = {'use_polarity': has_pol}
            # Read the architecture-defining params from config (highest priority
            # first). The model rebuilds its exact shape from these.
            for key in ('input_size', 'nb_filters', 'strides', 'kernel_size',
                        'dropout_rate', 'rrconv_iters', 'polarity_output_channels',
                        'pol_stream_width_mult', 'pol_init_rrconv_iters',
                        'pol_head_ps_att_width', 'pad_mode'):
                for src in (model_kwargs, ckpt_config, file_config):
                    if key in src:
                        rp90_kwargs[key] = src[key]
                        break
            # Fall back to the polarity head's output width if config is absent.
            if (has_pol and 'polarity_output_channels' not in rp90_kwargs
                    and 'polarity_head.weight' in state_dict):
                rp90_kwargs['polarity_output_channels'] = int(
                    state_dict['polarity_head.weight'].shape[0])
            model = build_mtan_r2unet_rp90_motion(**rp90_kwargs)
        else:
            model = MTAN_R2UNet(**model_kwargs)

        model.load_state_dict(state_dict)

        # Default pred_npts to the model's native window (config input_size[0])
        # so a 90 s checkpoint doesn't silently run with the 60 s default.
        if 'pred_npts' not in kwargs:
            isize = (model_kwargs.get('input_size') or ckpt_config.get('input_size')
                     or file_config.get('input_size'))
            if isinstance(isize, (list, tuple)) and isize:
                kwargs['pred_npts'] = int(isize[0])
            elif isinstance(isize, int):
                kwargs['pred_npts'] = isize

        predictor = cls(model=model, **kwargs)
        # Match inference normalization to how the checkpoint was TRAINED: a model trained
        # normalize_mode="moving" needs moving-normalize at inference; "zscore"
        # models keep the rolling default (the per-90s-window match for global z-score).
        _nm = file_config.get('normalize_mode') or ckpt_config.get('normalize_mode')
        if _nm == 'moving':
            predictor.long_trace_norm = 'moving'
            predictor.moving_filter_size = int(
                file_config.get('moving_filter_size', ckpt_config.get('moving_filter_size', 1024)))
        return predictor

    def _prepare_input(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Prepare waveform for model input.
        
        Args:
            waveform: (C, T) or (T, C) waveform array
        
        Returns:
            (1, C, T) normalized tensor
        """
        # Copy first: the per-channel normalization below writes in place, and
        # `.T` returns a view — without this the caller's array would be mutated.
        waveform = np.array(waveform, dtype=np.float32, copy=True)

        # Ensure (C, T) format
        if waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T

        # Normalize per channel
        for i in range(waveform.shape[0]):
            mean = np.mean(waveform[i])
            std = np.std(waveform[i])
            if std < 1e-8:
                std = 1.0
            waveform[i] = (waveform[i] - mean) / std
        
        # Handle NaN/Inf
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to tensor
        tensor = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)
        
        return tensor.to(self.device)
    
    @staticmethod
    def _build_flat_mask(
        wf_ct: np.ndarray,
        win_npts: int = 100,
        unique_tol: int = 5,
    ) -> np.ndarray:
        """
        Build a boolean mask (True = flat/constant region) over waveform samples.

        Scans every channel in non-overlapping windows of `win_npts` samples.
        A window is flagged as flat when the number of distinct floating-point
        values in *any* channel is <= unique_tol.  This reliably catches
        zero-filled, median-filled, or other constant boundary pads, while
        leaving genuine (even very quiet) seismic noise untouched.

        A channel that is constant over the WHOLE trace is skipped instead.
        That is a substituted channel, not a boundary pad: callers that supply
        three components fill an absent horizontal with zeros so the model gets
        the shape it expects. Scanning it would flag every window and zero the
        entire prediction, which reported no picks at all on records that still
        carry a usable vertical and one horizontal.

        Args:
            wf_ct:      (C, T) raw waveform, **before** any normalisation.
            win_npts:   Window length in samples (default 100 = 1 s at 100 Hz).
            unique_tol: Maximum distinct values to consider a window flat
                        (default 5).  Constant fills produce exactly 1 distinct
                        value.  Two limits are worth knowing.  A low-gain
                        channel whose noise is under one count also produces
                        3 to 5 distinct values per window and is flagged.  In
                        the other direction, a pad that has been through a
                        zero-phase filter is no longer constant: 1e-9 of
                        ringing is about 100 distinct values, so a caller that
                        filters before predicting loses this guard entirely.

        Returns:
            flat_mask: (T,) boolean array, True where the waveform is flat.
        """
        n_ch, n_samples = wf_ct.shape
        flat_mask = np.zeros(n_samples, dtype=bool)
        live = [ch for ch in range(n_ch)
                if len(np.unique(wf_ct[ch])) > unique_tol]
        if not live:
            # Every channel is constant. There is no signal to protect, so the
            # original all-flat answer is the right one.
            flat_mask[:] = True
            return flat_mask
        for start in range(0, n_samples, win_npts):
            end = min(start + win_npts, n_samples)
            for ch in live:
                if len(np.unique(wf_ct[ch, start:end])) <= unique_tol:
                    flat_mask[start:end] = True
                    break  # one flat channel is sufficient
        return flat_mask

    @torch.no_grad()
    def predict_array(
        self,
        waveform: np.ndarray,
        mode: Optional[str] = None,
        postprocess: bool = False,
        as_stream: bool = False,
        reference: Optional[Union[Stream, Trace]] = None,
        z_raw: Optional[np.ndarray] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Stream]:
        """
        Predict on a numpy array.

        z_raw: optional full-length (T,) raw vertical component for the polarity
        head's first-motion input. The model is trained on RAW Z, so when
        ``waveform`` is band-passed (for picker/detector) pass the unfiltered Z
        here. If None, the input's own Z channel is used (so a raw ``waveform``
        needs nothing extra). Affects polarity only — never picker/detector.

        mode selects how long waveforms are processed (default self.inference_mode):
          - "sliding": overlapping windows with tapered overlap-add (original
            RED-PAN style) + single-entry refinement. Smoother output.
          - "single": one forward pass per large block (leverages the model's
            length flexibility). Each sample predicted once; far cheaper.

        postprocess=True applies the event-level threshold gate (mean mask + peak
        P/S over each detected event must exceed the configured thresholds);
        rejected regions are zeroed and accepted events exposed via .last_events.

        Flat regions of the input waveform (zero-fill, median-fill, or any
        constant pad) are detected from the raw samples *before* normalisation
        and the corresponding prediction probabilities are zeroed out, suppressing
        false triggers at trace boundaries.

        Args:
            waveform: (C, T) or (T, C) waveform array

        as_stream=True instead returns one ObsPy Stream of the prediction
        vectors as probability traces (channels P, S, M=mask, and POL=signed
        first-motion U-D when the model has a polarity head); pass ``reference``
        (a Trace or Stream) to copy network/station/location/starttime onto the
        output traces. See ``to_stream``. .last_events / .last_polarity are still
        populated as side effects.

        Returns:
            (picker (T,3) P/S/Noise, detector (T,2) Mask/Unmask) by default, or
            an ObsPy Stream of P/S/M[/POL] traces when as_stream=True.
        """
        # Handle input shape
        if waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T  # -> (C, T)

        n_samples = waveform.shape[1]
        wf_orig = waveform  # keep raw reference for flat-region detection (before any padding)

        # Full-length raw vertical for the polarity stream. Falls back to the
        # input's Z channel (the pre-existing behaviour) when z_raw is None.
        if z_raw is None:
            z_full = waveform[self._ch_z]
        else:
            z_full = np.asarray(z_raw, dtype=np.float32).reshape(-1)
            if z_full.shape[0] != n_samples:
                raise ValueError(
                    f"z_raw length {z_full.shape[0]} != waveform length {n_samples}")

        mode = (mode or self.inference_mode).lower()
        self._last_events = None    # cleared each call; set only when postprocess=True
        self._last_polarity = None  # cleared each call; set by the predict paths below if a polarity head exists

        # Short waveform: direct single-window prediction (modes are identical).
        if n_samples <= self.pred_npts:
            if n_samples < self.pred_npts:
                pad_len = self.pred_npts - n_samples
                n_channels = waveform.shape[0]
                padded_channels = []
                for c in range(n_channels):
                    padded = np.pad(
                        waveform[c],
                        (0, pad_len),
                        mode='constant',
                        constant_values=0.0,
                    )
                    padded_channels.append(padded)
                waveform = np.stack(padded_channels, axis=0)
                z_full = np.pad(z_full, (0, pad_len), mode='constant', constant_values=0.0)

            x = self._prepare_input(waveform.copy())   # .copy() — _prepare_input normalises in place
            z_raw_arr = self._z_raw_from(z_full[None, :])   # raw, sign-preserved Z (polarity only)
            picker, detector, polarity = self._model_forward(x, z_raw_arr)
            picker = self._reorder_picker_output(picker)
            detector = self._reorder_detector_output(detector)
            picker   = picker[0].cpu().numpy().T[:n_samples]   # (T, 3)
            detector = detector[0].cpu().numpy().T[:n_samples]  # (T, 2)
            if polarity is not None:
                # (1, C, T) → (T, C); C=1 for signed [-1,+1], C=3 for softmax [N,U,D]
                self._last_polarity = polarity[0].cpu().numpy().T[:n_samples]
        elif mode == "single":
            # (2) Single forward pass per large block (length-flexible model).
            picker, detector = self._single_pass_predict(waveform, z_full)
        else:
            # (1) Original RED-PAN style: tapered overlap-add + single-entry refine.
            picker, detector = self._sliding_window_predict(waveform, z_full)

        # Zero out predictions where the raw waveform is flat (boundary fills)
        flat_mask = self._build_flat_mask(wf_orig)
        if flat_mask.any():
            picker[flat_mask]   = 0.0
            detector[flat_mask] = 0.0

        if postprocess:
            picker, detector = self._postprocess_threshold(
                picker, detector,
                mask_threshold=self.mask_threshold,
                p_threshold=self.p_threshold,
                s_threshold=self.s_threshold,
                detect_threshold=self.detect_threshold,
                smooth_npts=self.refine_smooth_npts,
            )

        if as_stream:
            return self.to_stream(picker, detector, reference=reference)
        return picker, detector

    def predict_arrays(
        self,
        waveform: np.ndarray,
        mode: Optional[str] = None,
        postprocess: bool = False,
        z_raw: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return (picker, detector, polarity) as an explicit 3-tuple.

        Identical to ``predict_array`` but surfaces the polarity head output as a
        return value instead of the hidden ``last_polarity`` side-effect. polarity
        is (T, 3) softmax over [N, U, D], or None when the model has no polarity
        head. ``z_raw`` (raw vertical for the polarity stream) is forwarded as-is.
        ``.last_events`` is still populated when ``postprocess=True``.
        """
        picker, detector = self.predict_array(
            waveform, mode=mode, postprocess=postprocess, z_raw=z_raw)
        return picker, detector, self._last_polarity

    def to_stream(
        self,
        picker: np.ndarray,
        detector: np.ndarray,
        reference: Optional[Union[Stream, Trace]] = None,
        polarity: Optional[np.ndarray] = None,
    ) -> Stream:
        """Wrap prediction vectors into one ObsPy Stream of probability traces.

        Channels: ``P`` = picker[:,0], ``S`` = picker[:,1], ``M`` = detector[:,0]
        (mask), and ``POL`` = signed first motion (U-D in [-1,1]) from the
        polarity head when available. network/station/location/starttime are
        copied from ``reference`` (a Trace or Stream, e.g. the vertical channel);
        sampling rate is 1/dt. ``polarity`` defaults to the last predict call's
        stored polarity.
        """
        ref = None
        if isinstance(reference, Stream):
            ref = reference[0].stats if len(reference) else None
        elif isinstance(reference, Trace):
            ref = reference.stats
        net = getattr(ref, "network", "") if ref else ""
        sta = getattr(ref, "station", "") if ref else ""
        loc = getattr(ref, "location", "") if ref else ""
        t0 = getattr(ref, "starttime", UTCDateTime(0)) if ref else UTCDateTime(0)

        chans = [("P", picker[:, 0]), ("S", picker[:, 1]), ("M", detector[:, 0])]
        pol = polarity if polarity is not None else self._last_polarity
        if pol is not None and pol.shape[0] == picker.shape[0] and pol.shape[1] >= 3:
            chans.append(("POL", pol[:, 1] - pol[:, 2]))   # signed first motion U-D

        st = Stream()
        for ch, data in chans:
            tr = Trace(data=np.ascontiguousarray(data, dtype=np.float32))
            tr.stats.network, tr.stats.station, tr.stats.location = net, sta, loc
            tr.stats.channel = ch
            tr.stats.starttime = t0
            tr.stats.sampling_rate = 1.0 / self.dt
            st.append(tr)
        return st

    def _sliding_window_predict(
        self,
        waveform: np.ndarray,
        z_full: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sliding window prediction for long waveforms, with automatic single-entry
        refinement.

        Steps:
          1. Boundary-pad the waveform with spectrum-matched noise (front) and
             per-channel median (back), each by one model window length.
          2. Overlap-add accumulation across all sliding windows.
          3. Trim padding to restore the original trace length.
          4. Single-entry refine: for each mask trigger, re-run one focused
             inference window and replace the trigger segment when the refined
             mask is stronger.
        """
        n_channels, n_samples = waveform.shape
        
        # Pad waveform at both ends to ensure edge coverage (spectrum-matched noise)
        pad_npts = self.pred_npts
        n_channels, _ = waveform.shape
        padded_channels = []
        for c in range(n_channels):
            ch = waveform[c]
            front_padded = pad_waveform_with_noise(ch, pad_npts, pad_position='front')
            back_padded = pad_waveform_with_noise(ch, pad_npts, pad_position='back')
            pad_front = front_padded[:pad_npts]
            pad_back = back_padded[-pad_npts:]
            padded_channels.append(np.concatenate([pad_front, ch, pad_back]))
        padded_waveform = np.stack(padded_channels, axis=0)
        padded_len = padded_waveform.shape[1]
        # Pad the raw-Z polarity input identically so its windows line up 1:1.
        z_front = pad_waveform_with_noise(z_full, pad_npts, pad_position='front')[:pad_npts]
        z_back = pad_waveform_with_noise(z_full, pad_npts, pad_position='back')[-pad_npts:]
        z_padded = np.concatenate([z_front, z_full, z_back])

        # Calculate window positions on padded waveform (as in the original RED-PAN)
        n_windows = (padded_len - self.pred_npts) // self.pred_interval_npts + 1
        window_starts = [i * self.pred_interval_npts for i in range(n_windows)]

        # Prepare windows (input + parallel raw-Z windows)
        windows = []
        z_windows = []
        for start in window_starts:
            windows.append(padded_waveform[:, start:start + self.pred_npts])
            z_windows.append(z_padded[start:start + self.pred_npts])
        
        # Batch predictions
        all_picker = []
        all_detector = []
        all_polarity = []

        for i in range(0, len(windows), self.batch_size):
            batch_windows = windows[i:i + self.batch_size]
            batch_z = z_windows[i:i + self.batch_size]

            # Normalize each window per channel (match training)
            batch_tensors = []
            for w in batch_windows:
                w_norm = w.copy()
                for c in range(w_norm.shape[0]):
                    mean = np.mean(w_norm[c])
                    std = np.std(w_norm[c])
                    if std < 1e-8:
                        std = 1.0
                    w_norm[c] = (w_norm[c] - mean) / std
                w_norm = np.nan_to_num(w_norm, nan=0.0, posinf=0.0, neginf=0.0)
                batch_tensors.append(w_norm)

            batch = torch.from_numpy(np.stack(batch_tensors, axis=0).astype(np.float32))
            batch = batch.to(self.device)

            # raw, sign-preserved Z for each window -> polarity stream
            z_raw_arr = self._z_raw_from(np.stack(batch_z, axis=0))
            picker, detector, polarity = self._model_forward(batch, z_raw_arr)
            picker = self._reorder_picker_output(picker)
            detector = self._reorder_detector_output(detector)

            all_picker.append(picker.cpu().numpy())
            all_detector.append(detector.cpu().numpy())
            if polarity is not None:
                all_polarity.append(polarity.cpu().numpy())

        all_picker = np.concatenate(all_picker, axis=0)  # (N, 3, T)
        all_detector = np.concatenate(all_detector, axis=0)  # (N, 2, T)
        has_polarity = len(all_polarity) > 0
        if has_polarity:
            all_polarity_np = np.concatenate(all_polarity, axis=0)  # (N, C, T) C=1 signed or C=3 softmax
            pol_channels = all_polarity_np.shape[1]

        # Overlap-add accumulation with position weights (tapered = RED-PAN style).
        # Use the actual channel counts from the model output rather than
        # hard-coding 3/2 — supports v8b's 1-channel sigmoid detector.
        picker_channels = all_picker[0].shape[0]
        detector_channels = all_detector[0].shape[0]
        picker_out = np.zeros((picker_channels, padded_len))
        detector_out = np.zeros((detector_channels, padded_len))
        if has_polarity:
            polarity_out = np.zeros((pol_channels, padded_len))
        weight_sum = np.zeros(padded_len)

        # Position weights: tapered (gaussian/cosine/triangular) = original
        # RED-PAN style; uniform = flat SeisBench-style. See self.window_weight.
        weights = self.position_weights

        for idx, start in enumerate(window_starts):
            end = start + self.pred_npts

            picker_out[:, start:end] += all_picker[idx] * weights
            detector_out[:, start:end] += all_detector[idx] * weights
            if has_polarity:
                polarity_out[:, start:end] += all_polarity_np[idx] * weights
            weight_sum[start:end] += weights

        # Normalize by weights
        weight_sum = np.maximum(weight_sum, 1e-8)
        picker_out /= weight_sum
        if has_polarity:
            polarity_out /= weight_sum
        detector_out /= weight_sum

        # Trim padding to original length  →  (T, C)
        picker_tc   = picker_out[:, pad_npts:pad_npts + n_samples].T
        detector_tc = detector_out[:, pad_npts:pad_npts + n_samples].T
        if has_polarity:
            # (T, 3) — channels are [N, U, D]; argmax over axis=1 gives the class
            self._last_polarity = polarity_out[:, pad_npts:pad_npts + n_samples].T

        # Single-entry refine pass on the original (C, T) waveform
        picker_tc, detector_tc = self._single_entry_refine(
            waveform, z_full, picker_tc, detector_tc,
            pre_trigger_sec=self.refine_pre_trigger_sec,
            trigger_on=self.refine_trigger_on,
            trigger_off=self.refine_trigger_off,
            smooth_npts=self.refine_smooth_npts,
        )

        return picker_tc, detector_tc

    def _single_pass_predict(
        self,
        waveform: np.ndarray,
        z_full: np.ndarray,
        block_npts: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-forward-pass prediction for long waveforms.

        Leverages the model's built-in length flexibility: each block is run
        through the network in ONE forward pass — no overlapping windows, no
        overlap-add. Every sample is predicted exactly once, so this is far
        cheaper than sliding mode, at the cost of no seam blending between blocks
        and per-block (rather than per-90 s-window) normalisation.

        A whole day fed as a single tensor would exhaust GPU memory, so the trace
        is processed in non-overlapping blocks of ``block_npts`` samples (default
        ``self.single_pass_block``) and concatenated. If the trace fits in one
        block this is literally a single forward pass.
        """
        n_channels, n_samples = waveform.shape
        block = max(int(block_npts or self.single_pass_block), self.pred_npts)

        # Rolling (per-window) normalization over the WHOLE trace, once: each sample is
        # normalized by a centered ``rolling_window`` moving mean/std, so a big earthquake
        # only inflates its own ~window neighbourhood instead of a whole 600s block —
        # matching the per-90s training-window z-score continuously. "global" = legacy.
        wf_src = None
        if self.long_trace_norm in ("rolling", "moving"):
            from scipy.ndimage import uniform_filter1d
            wf32 = waveform.astype(np.float32)
            if self.long_trace_norm == "moving":
                # PhaseNet+ style: subtract a moving mean, divide by the moving L1 (mean-abs)
                # over a ~10 s window — matches normalize_mode="moving" training (EQNet/dataset_v2).
                win = int(self.moving_filter_size)
                d = wf32 - uniform_filter1d(wf32, win, axis=-1, mode="reflect")
                mabs = uniform_filter1d(np.abs(d), win, axis=-1, mode="reflect")
                wf_src = (d / np.where(mabs > 1e-8, mabs, 1.0)).astype(np.float32)
            else:  # rolling: centered pred_npts window, L2 (mean/std) — matches zscore training
                win = int(self.rolling_window)
                m = uniform_filter1d(wf32, win, axis=-1, mode="reflect")
                v = uniform_filter1d(wf32 * wf32, win, axis=-1, mode="reflect") - m * m
                wf_src = ((wf32 - m) / (np.sqrt(np.maximum(v, 0.0)) + 1e-6)).astype(np.float32)

        picker_parts, detector_parts = [], []
        polarity_parts = []
        has_pol = False

        for start in range(0, n_samples, block):
            end = min(start + block, n_samples)
            seg = waveform[:, start:end]
            seg_len = seg.shape[1]
            if wf_src is not None:
                seg_norm = wf_src[:, start:end].copy()   # already rolling-normalized
            else:
                # legacy per-block global z-score (seg is (C, seg_len) — avoid
                # _prepare_input's transpose heuristic for short tail blocks).
                seg_norm = seg.astype(np.float32, copy=True)
                for ci in range(seg_norm.shape[0]):
                    s = float(seg_norm[ci].std())
                    seg_norm[ci] = (seg_norm[ci] - float(seg_norm[ci].mean())) / (s if s > 1e-8 else 1.0)
            seg_norm = np.nan_to_num(seg_norm, nan=0.0, posinf=0.0, neginf=0.0)
            x = torch.from_numpy(seg_norm).unsqueeze(0).to(self.device)   # (1, C, seg_len)
            z_raw_arr = self._z_raw_from(z_full[start:end][None, :])      # raw, sign-preserved Z (polarity)
            picker, detector, polarity = self._model_forward(x, z_raw_arr)
            picker = self._reorder_picker_output(picker)
            detector = self._reorder_detector_output(detector)
            picker_parts.append(picker[0].cpu().numpy().T[:seg_len])      # (seg_len, 3)
            detector_parts.append(detector[0].cpu().numpy().T[:seg_len])  # (seg_len, 2)
            if polarity is not None:
                has_pol = True
                polarity_parts.append(polarity[0].cpu().numpy().T[:seg_len])

        picker_tc = np.concatenate(picker_parts, axis=0)
        detector_tc = np.concatenate(detector_parts, axis=0)
        n_blocks = len(picker_parts)
        self._last_polarity = (np.concatenate(polarity_parts, axis=0)
                               if has_pol and len(polarity_parts) == n_blocks else None)
        return picker_tc, detector_tc

    def _postprocess_threshold(
        self,
        picker: np.ndarray,
        detector: np.ndarray,
        mask_threshold: float,
        p_threshold: float,
        s_threshold: float,
        detect_threshold: float,
        smooth_npts: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Event-level threshold gate.

        Delineates candidate events from the detector mask (``trigger_onset`` at
        ``detect_threshold``), then KEEPS an event only when ALL hold over the
        event span: mean mask probability >= ``mask_threshold``, peak P
        probability >= ``p_threshold``, peak S probability >= ``s_threshold``.
        Samples outside accepted events are zeroed in picker, detector and the
        stored polarity. Accepted events are exposed via ``.last_events``.

        Set ``s_threshold`` (or ``p_threshold``) to 0.0 to drop that phase's
        requirement (e.g. to keep P-only detections).
        """
        from obspy.signal.trigger import trigger_onset

        mask = detector[:, 0]
        if smooth_npts and smooth_npts > 1:
            kernel = np.ones(smooth_npts) / smooth_npts
            m_smooth = np.convolve(mask, kernel, mode="same")
        else:
            m_smooth = mask

        # predict_array returns the picker as [P, S, Noise] (see its docstring),
        # so P is channel 0 and S is channel 1.
        p_idx, s_idx = 0, 1

        triggers = trigger_onset(m_smooth, detect_threshold, detect_threshold)
        sp = self.sp_thresholds
        keep = np.zeros(len(mask), dtype=bool)
        events = []
        for on, off in triggers:
            on, off = int(on), int(off)
            if off <= on:
                continue
            seg_p = picker[on:off + 1, p_idx]
            seg_s = picker[on:off + 1, s_idx]
            mean_mask = float(mask[on:off + 1].mean())
            peak_mask = float(mask[on:off + 1].max())
            peak_p = float(seg_p.max())
            peak_s = float(seg_s.max())
            if sp is not None:
                # S-P-adaptive: trigger DURATION -> S-P bin -> (mask-MEAN, P, S) thresholds.
                # Gate on the mask MEAN over mask[on:off] and the max P/S over the same
                # window == benchmark extract_triggers (the trigger is already smoothed +
                # delineated at detect_threshold=0.1, so this matches the fit calibration).
                m_thr, p_thr, s_thr = _sp_params_for_duration((off - on) * self.dt, sp)
                accept = (float(mask[on:off].mean()) >= m_thr
                          and float(picker[on:off, p_idx].max()) >= p_thr
                          and float(picker[on:off, s_idx].max()) >= s_thr)
            else:
                accept = (mean_mask >= mask_threshold and peak_p >= p_threshold
                          and peak_s >= s_threshold)
            if accept:
                keep[on:off + 1] = True
                events.append({
                    "on": on, "off": off,
                    "mean_mask": mean_mask, "peak_mask": peak_mask,
                    "peak_p": peak_p, "peak_s": peak_s,
                    "p_index": on + int(seg_p.argmax()),
                    "s_index": on + int(seg_s.argmax()),
                })

        picker_g = picker.copy()
        detector_g = detector.copy()
        picker_g[~keep] = 0.0
        detector_g[~keep] = 0.0
        if (self._last_polarity is not None
                and self._last_polarity.shape[0] == len(mask)):
            pol = self._last_polarity.copy()
            pol[~keep] = 0.0
            self._last_polarity = pol
        self._last_events = events
        return picker_g, detector_g

    def _single_entry_refine(
        self,
        wf_ct: np.ndarray,
        z_full: np.ndarray,
        picker_out: np.ndarray,
        detector_out: np.ndarray,
        pre_trigger_sec: float = 5.0,
        trigger_on: float = 0.3,
        trigger_off: float = 0.3,
        smooth_npts: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-entry refinement pass after sliding-window prediction.

        For each mask trigger detected in the sliding output, one focused
        inference window is run (starting pre_trigger_sec before the trigger
        onset) and the trigger segment is replaced when the refined mask mean
        probability is >= the original.

        Args:
            wf_ct:        (C, T) channel-first waveform, not pre-normalised.
            z_full:       (T,) raw vertical for the polarity stream (windowed in
                          sync with wf_ct).
            picker_out:   (T, 3) picker output from sliding pass.
            detector_out: (T, 2) detector output from sliding pass.
            pre_trigger_sec: How many seconds before trigger onset to start the
                             focused window (default 5 s).
            trigger_on/off:  Mask thresholds for trigger_onset (default 0.3).
            smooth_npts:     Moving-average kernel length for mask smoothing.

        Returns:
            Refined (picker_out, detector_out) as copies.
        """
        from obspy.signal.trigger import trigger_onset

        n_channels, n_samples = wf_ct.shape
        wf_tc = wf_ct.T  # (T, C) view — sliced below as copies

        # Mask channel is always index 0 of detector_out (signal channel)
        m_arr = detector_out[:, 0]

        # Smooth mask and detect triggers
        if smooth_npts > 1:
            kernel = np.ones(smooth_npts) / smooth_npts
            m_smooth = np.convolve(m_arr, kernel, mode='same')
        else:
            m_smooth = m_arr

        triggers = trigger_onset(m_smooth, trigger_on, trigger_off)
        if len(triggers) == 0:
            return picker_out, detector_out

        pre_npts = max(0, int(round(pre_trigger_sec / self.dt)))
        ref_picker   = picker_out.copy()
        ref_detector = detector_out.copy()

        for on_idx, off_idx in triggers:
            on_idx, off_idx = int(on_idx), int(off_idx)
            if off_idx <= on_idx:
                continue

            # Position focused window: start pre_npts before trigger onset
            win_start = max(0, on_idx - pre_npts)
            win_end   = win_start + self.pred_npts
            if win_end > n_samples:
                win_start = max(0, n_samples - self.pred_npts)
                win_end   = n_samples

            wf_win = wf_tc[win_start:win_end].copy()  # (win_len, C)
            z_win = z_full[win_start:win_end].copy()  # (win_len,) parallel raw Z (polarity)
            if len(wf_win) < self.pred_npts:
                pad_n = self.pred_npts - len(wf_win)
                wf_win = np.concatenate(
                    [wf_win, np.zeros((pad_n, n_channels), dtype=np.float32)], axis=0)
                z_win = np.concatenate([z_win, np.zeros(pad_n, dtype=np.float32)])

            # Single forward pass on the focused window (raw-Z polarity input)
            try:
                with torch.no_grad():
                    wf_win_ct = wf_win.T  # (C, T)
                    x = self._prepare_input(wf_win_ct.copy())  # normalised (1, C, T)
                    z_raw_arr = self._z_raw_from(z_win[None, :])   # raw, sign-preserved Z (polarity)
                    p_raw, d_raw, pol_raw = self._model_forward(x, z_raw_arr)
                    p_raw = self._reorder_picker_output(p_raw)[0].cpu().numpy().T    # (T, 3)
                    d_raw = self._reorder_detector_output(d_raw)[0].cpu().numpy().T  # (T, 2)
                    pol_raw = (pol_raw[0].cpu().numpy().T
                               if pol_raw is not None else None)               # (T, C)
            except (RuntimeError, ValueError) as exc:
                warnings.warn(f"single-entry refine failed at {on_idx}-{off_idx}: {exc}")
                continue

            # Replace trigger region [rep_st:rep_ed] when refined mask is stronger
            rep_st   = max(0, on_idx)
            rep_ed   = min(n_samples, off_idx)
            if rep_ed <= rep_st:
                continue
            local_st = rep_st - win_start
            local_ed = local_st + (rep_ed - rep_st)
            if local_st < 0 or local_ed > self.pred_npts:
                continue

            old_mean = float(np.mean(m_arr[rep_st:rep_ed]))
            new_mean = float(np.mean(d_raw[local_st:local_ed, 0]))
            if new_mean >= old_mean:
                ref_picker[rep_st:rep_ed]   = p_raw[local_st:local_ed]
                ref_detector[rep_st:rep_ed] = d_raw[local_st:local_ed]
                # keep the stored polarity consistent with the refined picks
                if (pol_raw is not None and self._last_polarity is not None
                        and self._last_polarity.shape[0] == n_samples):
                    self._last_polarity[rep_st:rep_ed] = pol_raw[local_st:local_ed]

        return ref_picker, ref_detector

    def _reorder_picker_output(self, picker: torch.Tensor) -> torch.Tensor:
        """Reorder picker output to desired channel order."""
        if self.picker_output_permutation is not None and picker.dim() == 3:
            return picker[:, list(self.picker_output_permutation), :]
        if self.output_order == 'NPS':
            return picker
        if picker.dim() != 3:
            return picker
        # (B, C, T) -> reorder channels
        return picker[:, [1, 2, 0], :]

    def _reorder_detector_output(self, detector: torch.Tensor) -> torch.Tensor:
        """Reorder detector output to desired channel order."""
        if self.detector_output_permutation is not None and detector.dim() == 3:
            return detector[:, list(self.detector_output_permutation), :]
        return detector
    
    def predict(
        self,
        wf: Stream,
        postprocess: bool = False,
        mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on an ObsPy Stream (matching TF API).

        Args:
            wf: ObsPy Stream with 3-component seismic data
            postprocess: Apply the event-level threshold gate (see predict_array)
            mode: "sliding" (RED-PAN tapered overlap-add) or "single" (one
                forward pass per block); defaults to self.inference_mode

        Returns:
            Tuple of (picker_predictions, detector_predictions)
        """
        # Extract data from stream
        wf = wf.copy()
        wf.sort()

        # Ensure 3 channels (pad missing with zeros)
        if len(wf) < 3:
            for _ in range(3 - len(wf)):
                tr = wf[0].copy()
                tr.data = np.zeros_like(tr.data)
                wf.append(tr)
        wf = wf[:3]

        # Ensure equal length with noise padding
        wf = sac_len_complement(wf, max_length=None, pad_mode='noise')

        # Get waveform data (3, T)
        data = np.stack([tr.data.astype(np.float32) for tr in wf[:3]], axis=0)
        
        picker, detector = self.predict_array(
            data, mode=mode, postprocess=postprocess)

        return picker, detector

    @property
    def last_polarity(self) -> Optional[np.ndarray]:
        """Polarity output from the last predict/predict_array call.

        Returns (T, 3) softmax probabilities over [N, U, D] channels (PhaseNet+
        style), or None if model has no polarity head. Use ``argmax(axis=-1)``
        with a confidence threshold (e.g. >0.33 on U/D) to obtain discrete
        polarity calls. Access after calling predict() or predict_array().
        """
        return self._last_polarity

    @property
    def last_events(self) -> Optional[list]:
        """Accepted events from the last postprocess gate, or None.

        List of dicts {on, off, mean_mask, peak_p, peak_s, p_index, s_index}
        (sample indices). Populated only when predict/predict_array was called
        with postprocess=True.
        """
        return self._last_events
