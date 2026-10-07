"""The RED-PAN models scored in the manuscript, loaded from the ``redpan_motion`` package.

Model keys used by every entry point and in the results layout:

    edge           Edge-RED-PAN-Motion  (shipped checkpoint ``edge_rp90``)        1 Hz high-pass input
    rpm            RED-PAN-Motion       (shipped checkpoint ``redpan_motion``)    1 Hz high-pass input
    redpan         RED-PAN, 2022 paper model (``redpan_paper_ckpt``, see README)  1-45 Hz band-pass input
    redpan_240107  RED-PAN release REDPAN_60s_240107 (shipped ``redpan_60s``)     1-45 Hz band-pass input

The returned handle exposes what the runners call: ``.backbone.inner`` (the network),
``.in_samples`` (6000 for the 60 s models, 9000 otherwise) and ``.is_redpan60``.
The shipped edge_rp90 / redpan_motion weights are identical to the ones benchmarked for the
manuscript (checked tensor by tensor against the SeisBench-format copies used then).
"""

from __future__ import annotations

import torch
from torch import nn

FILTER = {"edge": "hp1", "rpm": "hp1", "redpan": "bp145", "redpan_240107": "bp145"}
MODEL_KEYS = tuple(FILTER)
# CSV ``model`` column / file prefix per key (kept stable for the scorers)
LABEL = {
    "edge": "EDGE_RPM",
    "rpm": "RP_MOTION",
    "redpan": "REDPAN_60S_PAPER",
    "redpan_240107": "REDPAN_60S_240107",
}
SIXTY_S = ("redpan", "redpan_240107")  # the 60 s RED-PAN models (6,000-sample window, no polarity)


class ModelHandle(nn.Module):
    """The network (``.backbone.inner``), its window length and whether it is a 60 s RED-PAN."""

    def __init__(self, net: nn.Module, in_samples: int, is_redpan60: bool):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.inner = net
        self.in_samples = in_samples
        self.is_redpan60 = is_redpan60


def checkpoint_for(key: str, cfg: dict) -> str:
    """Shipped checkpoint name of a model key, or the configured path of the 2022 paper model."""
    return {
        "edge": "edge_rp90",
        "rpm": "redpan_motion",
        "redpan_240107": "redpan_60s",
        "redpan": str(cfg["redpan_paper_ckpt"]),
    }[key]


def load_model(key: str, cfg: dict, device: str = "cpu") -> ModelHandle:
    """Build the network from its checkpoint + sibling config.json, in eval mode."""
    if key not in FILTER:
        raise KeyError(f"unknown model {key!r}; choose from {MODEL_KEYS}")
    from redpan_motion.inference import REDPANPredictor

    net = REDPANPredictor.from_checkpoint(checkpoint_for(key, cfg), device="cpu").model
    is60 = key in SIXTY_S
    h = ModelHandle(net, 6000 if is60 else 9000, is60)
    h.eval()
    return h.to(device)


def default_device() -> str:
    """``cuda:0`` when a GPU is visible, else ``cpu``."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"
