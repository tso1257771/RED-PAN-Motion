"""S-P-time-adaptive joint detection thresholds.

The deploy API detects in two stages — the detector MASK must trigger, then the max P
and max S pick-probabilities are read INSIDE that triggered window. The optimal
``(mask_mean, P, S)`` triple depends on the event's S-P time (a distance proxy: the S
goes weak/emergent at range), so this module holds, per model, the S-P-binned thresholds
plus a fallback default, and maps a measured mask-trigger DURATION to the right bin.

Why mask duration: the detector mask is trained to wrap P -> S (+ boundary), so the
trigger width is a direct S-P proxy (corr ~0.64; mask_dur ~= S-P + ~2 s). No second
S-search pass is needed — the width you already have from the trigger delineates the bin.

MASK GATE = MEAN, not peak. The mask threshold is compared against the MEAN mask over the
trigger window (``mask[on:off]``), not its peak. For long-S-P events the trigger is wide
(it wraps P->S), so a peak gate waves through brief noise spikes inside the window while a
mean gate demands SUSTAINED detection, which rejects brief noise inside a long trigger.
The trigger MUST be delineated exactly as it was when the thresholds were fitted: smoothed mask
(``np.convolve`` mean kernel, npts=``_SP_SMOOTH_NPTS``) with ``trigger_onset`` at
``_SP_TRIG_ONSET`` on/off — the mean is window-sensitive, so the thresholds only transfer
under this recipe. ``detect_events_joint`` and ``_postprocess_threshold`` both use it.

Thresholds were fit by a joint (mask-mean-gated) sweep over all 8 test datasets with the
EXACT inside-mask JOINT noise false-positive (a noise trace counts only if mask-mean AND
inside-max-P AND inside-max-S all fire, read only WITHIN each trigger). Fit data is strictly
held out from training: EQ = the 8 SeisBench TEST splits (incl. ``crew``, never in training
at all); noise = 5 clean held-out pools (STEAD/GeoNet/INSTANCE/RockNet/TW; STEAD uses its
val split — its noise has no test split). ``CEED_val`` is excluded: CEED has no noise
category, its "drop_noise" traces are labelled earthquakes (P & S picks). Update ``_TABLES``
in one place if re-fit.

.. warning::
   The ``redpan_60s`` table is PROVISIONAL as of 0.1.3. It was fit against the
   previous ``redpan_60s`` checkpoint, which was converted from a different
   TensorFlow run (RP60_03). The checkpoint was rebuilt in 0.1.3 from
   ``REDPAN_60s_240107``, the RED-PAN release of 2024-01-07, and these thresholds have not
   been re-fit against it. The ``redpan_motion`` and ``edge_rp90`` tables are
   unaffected.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

# Ordered S-P bins (seconds). Index i <-> _SP_BIN_LABELS[i].
_SP_BIN_LABELS: Tuple[str, ...] = ("[0,5)", "[5,10)", "[10,15)", "[15,20)", "[20,+)")

# Mask-trigger-duration (s) upper edges -> S-P bin index (validated reverse mapping:
# dur<6 -> S-P~3s, 6-12 -> ~8s, 12-18 -> ~13s, 18-24 -> ~20s, >=24 -> ~29s).
_DUR_EDGES: Tuple[float, ...] = (6.0, 12.0, 18.0, 24.0)

# Trigger delineation the mean-gate thresholds are calibrated on: smooth the mask with a length-_SP_SMOOTH_NPTS moving average, then
# trigger_onset at _SP_TRIG_ONSET (on == off). The mask MEAN is taken over mask[on:off].
_SP_SMOOTH_NPTS: int = 10
_SP_TRIG_ONSET: float = 0.1

# Per model: 5 S-P-bin (mask_MEAN, P, S) triples + a fallback default triple.
# mask_MEAN = threshold on the MEAN mask over the trigger (see module docstring), NOT peak.
# default = conservative fallback used only when no trigger duration is available (rare —
# every delineated trigger has a duration and so selects a bin directly).
_TABLES: Dict[str, Dict[str, Union[Tuple[float, float, float], List[Tuple[float, float, float]]]]] = {
    "edge_rp90": {
        "default": (0.80, 0.20, 0.20),
        "bins": [(0.60, 0.30, 0.20), (0.70, 0.20, 0.20), (0.80, 0.10, 0.20),
                 (0.90, 0.10, 0.10), (0.90, 0.10, 0.10)],
    },
    "rp90_motion_v49": {
        "default": (0.80, 0.20, 0.20),
        "bins": [(0.60, 0.50, 0.20), (0.80, 0.20, 0.20), (0.90, 0.20, 0.20),
                 (0.90, 0.20, 0.20), (0.90, 0.10, 0.20)],
    },
    "redpan_60s": {
        "default": (0.80, 0.20, 0.20),
        "bins": [(0.60, 0.40, 0.20), (0.80, 0.10, 0.20), (0.90, 0.20, 0.20),
                 (0.90, 0.20, 0.20), (0.90, 0.20, 0.20)],
    },
}

# model_type / class-name aliases -> canonical table key
_ALIASES: Dict[str, str] = {
    "edge_rp90": "edge_rp90", "edge_rp90_v1": "edge_rp90", "edgerp90": "edge_rp90",
    "rp90_motion_v49": "rp90_motion_v49", "mtan_r2unet_rp90_motion": "rp90_motion_v49",
    "rp90_motion": "rp90_motion_v49", "v49": "rp90_motion_v49",
    "redpan_60s": "redpan_60s", "redpan60s": "redpan_60s", "redpan_tf60": "redpan_60s",
}

ThreshTriple = Tuple[float, float, float]


def available_models() -> Tuple[str, ...]:
    """Canonical model keys with a threshold table."""
    return tuple(_TABLES)


def sp_bin_from_duration(duration_sec: float) -> int:
    """Map a mask-trigger duration (s) to an S-P bin index (0..4)."""
    for i, edge in enumerate(_DUR_EDGES):
        if duration_sec < edge:
            return i
    return len(_DUR_EDGES)


def sp_bin_label(duration_sec: float) -> str:
    """Human-readable S-P bin for a mask-trigger duration."""
    return _SP_BIN_LABELS[sp_bin_from_duration(duration_sec)]


def resolve_table(spec: Union[None, bool, str, Dict]) -> Optional[Dict]:
    """Resolve an ``sp_adaptive_thresholds`` argument to a table dict (or None = off).

    ``spec`` may be: None/False (off); a model key or model_type/class name (str);
    or a table dict with 'default' + 'bins' (5 triples) to use verbatim.
    """
    if spec is None or spec is False:
        return None
    if isinstance(spec, dict):
        if "bins" in spec and "default" in spec:
            _validate(spec)
            return spec
        raise ValueError("custom sp_adaptive_thresholds dict needs 'default' and 'bins' (5 triples)")
    if isinstance(spec, str):
        key = _ALIASES.get(spec.lower(), spec)
        if key not in _TABLES:
            raise KeyError(f"no S-P threshold table for {spec!r}; known: {available_models()}")
        return _TABLES[key]
    raise TypeError(f"sp_adaptive_thresholds must be None/str/dict, got {type(spec).__name__}")


def params_for_duration(duration_sec: float, table: Dict) -> ThreshTriple:
    """(mask_mean, P, S) thresholds for a mask-trigger duration, from a resolved table."""
    return tuple(table["bins"][sp_bin_from_duration(duration_sec)])  # type: ignore[return-value]


def default_params(table: Dict) -> ThreshTriple:
    """Fallback (mask_mean, P, S) when S-P is unknown."""
    return tuple(table["default"])  # type: ignore[return-value]


def floor_thresholds(table: Dict) -> ThreshTriple:
    """Lowest (P, S) over all bins + default — the candidate-peak floor so per-trigger
    adaptive thresholds only ever RAISE the bar. Returns (min_mask, min_P, min_S)."""
    triples = list(table["bins"]) + [table["default"]]
    return (min(t[0] for t in triples), min(t[1] for t in triples), min(t[2] for t in triples))


def _validate(table: Dict) -> None:
    if len(table["bins"]) != len(_SP_BIN_LABELS):
        raise ValueError(f"table 'bins' must have {len(_SP_BIN_LABELS)} triples")
    for t in list(table["bins"]) + [table["default"]]:
        if len(t) != 3:
            raise ValueError("each threshold entry must be (mask, P, S)")


for _t in _TABLES.values():  # fail fast on a malformed built-in table
    _validate(_t)
