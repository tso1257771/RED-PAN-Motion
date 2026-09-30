# RED-PAN-Motion

Multi-task seismic deep learning for **90-second** waveform windows. A single
MTAN R2U-Net performs three tasks at once:

- **Detection**: a per-sample earthquake / no-earthquake mask
- **Phase picking**: P / S / Noise probabilities
- **First-motion polarity**: Up / Down / None at the P arrival

This is the standalone, **pure-PyTorch** home of the RP90-Motion model
(checkpoint `redpan_motion`). The original TensorFlow RED-PAN is at
[tso1257771/RED-PAN](https://github.com/tso1257771/RED-PAN).

---

## Why 90 s?

The 90-second window (9000 samples at 100 Hz) covers large events whose P to S
separation the 60-second RED-PAN truncates. The depth-5 backbone keeps single
pass inference fast on long continuous traces and adds a polarity head for
first motion.

## Install

The package includes the three pretrained checkpoints, so one line installs
everything needed to run them:

```bash
pip install "redpan_motion[seisbench] @ git+https://github.com/tso1257771/RED-PAN-Motion@v0.1.3"
```

For development, from a clone:

```bash
git clone https://github.com/tso1257771/RED-PAN-Motion
cd RED-PAN-Motion
pip install -e .            # core (pure PyTorch)
pip install -e ".[dev]"     # + pytest / black / ruff
pip install -e ".[seisbench]"  # + seisbench, for redpan_motion.integrations.seisbench
pip install -e ".[tf]"      # + tensorflow, only for TF->torch weight conversion
```

Core dependencies are `torch>=2.0, numpy, scipy, pandas, obspy, h5py, tqdm`.
**No TensorFlow is imported at runtime.** The few `import tensorflow` calls are
lazy and live only inside the optional helpers that convert weights.

## Quick start

`REDPANPredictor.from_checkpoint` takes the name of a shipped checkpoint
(`redpan_60s`, `redpan_motion` or `edge_rp90`) or a path to a `.pt` file. It
reads the `config.json` beside the weights and rebuilds the architecture the
checkpoint was trained with.

```python
import numpy as np
from redpan_motion.inference import REDPANPredictor

predictor = REDPANPredictor.from_checkpoint("redpan_motion", device="cuda")

# Sliding-window inference over a long 3-component trace (C, T) at 100 Hz.
waveform = np.random.randn(3, 60000).astype("float32")
picker, detector = predictor.predict_array(waveform)   # (T, 3) P/S/N ; (T, 2) event mask
polarity = predictor.last_polarity                     # (T, 3) N/U/D (None, Up, Down), aligned to picker (property)
```

A single 90 s window straight through the network returns the raw 3-tuple:

```python
import torch
model = predictor.model.eval()                         # MTAN_R2UNet_RP90_Motion
with torch.no_grad():
    picker, polarity, detector = model(torch.randn(1, 3, 9000))
# picker (1,3,9000) P/S/N · polarity (1,3,9000) N/U/D · detector (1,2,9000) event mask
```

> `scripts/verify_install.py` shows the explicit
> `build_mtan_r2unet_rp90_motion(**config)` construction, for use without the
> predictor.

### Through SeisBench

With the `seisbench` extra installed, the three checkpoints load as SeisBench
`WaveformModel`s, so `annotate` and `classify` work on ObsPy streams as they do
for PhaseNet or EQTransformer:

```python
from redpan_motion.integrations.seisbench import RedpanSB60s, RedpanSB90s

model = RedpanSB90s.from_redpan_checkpoint("redpan_motion")  # or "edge_rp90"
# model = RedpanSB60s.from_redpan_checkpoint("redpan_60s")
picks = model.classify(stream).picks      # stream: an obspy.Stream at any sampling rate
```

### Worked example

[`notebooks/fdsn_inference.ipynb`](notebooks/fdsn_inference.ipynb) downloads the
waveforms of one earthquake from FDSN services with ObsPy, runs the three
checkpoints through both interfaces, and compares the picks with iasp91 arrival
times.

## Pretrained checkpoints

Three checkpoints ship inside the package, in `redpan_motion/checkpoints/`,
so an install from GitHub includes them. Each directory holds the best epoch
as `best.pt` together with the `config.json` needed to rebuild its
architecture. The loaders take the directory name; a bare name always means
the shipped checkpoint, and `redpan_motion.checkpoints.checkpoint_dir(name)`
gives its path.

| checkpoint | window | heads | params | notes |
|---|---|---|---:|---|
| `redpan_60s` | 60 s / 6000 | picker, detector | ~350 k | The original RED-PAN 60 s model of Liao et al. (2022), converted to PyTorch from its TensorFlow weights by `scripts/convert_redpan_60s.py`. On random input its output differs from the TensorFlow model by at most about 2e-7. It has no polarity head. |
| `redpan_motion` | 90 s / 9000 | picker, detector, polarity | ~544 k | `MTAN_R2UNet_RP90_Motion` (v49), backbone `[8,16,24,32,40]`, trained from scratch on nine datasets: TW, CEED_NC, CEED_SC, INSTANCE, GeoNet, OBSTransformer, ROMPLUS, RockNet and STEAD. |
| `edge_rp90` | 90 s / 9000 | picker, detector, polarity | ~309 k | `EdgeRP90`, 0.205 GMAC per window, about 4.5 times fewer operations than `redpan_motion`. Trained with the same data and recipe. ONNX exports, fp32 and int8, are in `edge_model_design/deploy/`. |

Parameter counts are printed by `scripts/verify_install.py`. The tests check
the exact count of `redpan_60s` and the parameter and MAC budgets of
`edge_rp90`.

`redpan_motion/checkpoints/edge_rp90/final.pt` holds the weights of the last
training epoch as a bare state dict. The released model is `best.pt`
(epoch 189). `final.pt` is in the repository only, not in the installed
package, and nothing loads it.

## Architecture

`MTAN_R2UNet_RP90_Motion` is a depth-5 MTAN (Multi-Task Attention Network) over
a Recurrent-Residual U-Net (R2U-Net) backbone. Inference accepts any input
length `T`, which is padded to the encoder stride and cropped back.

- **Input** `(B, 3, 9000)`, the E, N and Z components, in that order, at 100 Hz. Raw Z is kept
  un-normalized for the polarity head so that the sign of first motion is
  preserved.
- **Encoder and decoder**: 5 levels, `nb_filters` per level, `strides`
  `[5,5,3,3,2]`, kernel size 7, RR-conv iterations 2, reflect padding.
- **Heads**: a shared backbone with task-attention branches for the picker
  (P/S/N), the detector (event mask), and a gated polarity stream
  (`pol_stream_width_mult`).
- **Multi-task training**: Dynamic Weight Averaging (DWA) balances the picker,
  detector and polarity losses. Mixed precision (AMP) is supported.

## Project layout

```
RED-PAN-Motion/
├── redpan_motion/               # the package
│   ├── models/             # mtan_r2unet (base blocks), mtan_r2unet_rp90_motion,
│   │                       #   edge_rp90, redpan_60s
│   ├── inference/          # predictor.py, REDPANPredictor (single/sliding, to_stream)
│   ├── picks.py            # the redpan_picks DataFrame from model output
│   ├── pairing.py          # mask-detection P-S pairing
│   ├── sp_thresholds.py    # detection thresholds that vary with S-P time
│   ├── signal.py           # the model-input filters: highpass, bandpass
│   ├── waveform_io.py      # SAC and StationXML reading, grouping of a day archive by instrument
│   ├── checkpoints/        # the released weights: redpan_60s, redpan_motion, edge_rp90
│   ├── integrations/       # seisbench.py: the three checkpoints as SeisBench models
│   ├── amplitudes.py       # from RED-PAN: amplitudes, Wood-Anderson, SNR
│   ├── response.py         # from RED-PAN: sensor and response constants
│   ├── training/           # REDPANTrainer, losses (DWA)
│   ├── data/               # dataset.py, dataset_v2.py (MultiDatasetH5), augmentation
│   └── utils/              # waveform padding, optimization, weight conversion
├── scripts/                # daily_inference, picks_to_csv, template_inference,
│   │                       #   train_rp90_motion, verify_install, convert_redpan_60s
│   └── benchmarks/         #   benchmark_stead, benchmark_ceed_polarity,
│                           #   benchmark_polarity_filt, benchmark_p_pick_timing
├── notebooks/              # fdsn_inference.ipynb (FDSN download and picking),
│                           #   daily_inference.py (a day archive, as Jupyter cells)
├── edge_model_design/deploy/    # edge_rp90 exported to ONNX, fp32 and INT8
├── configs/                # training configs
├── tests/                  # pytest suite
├── pyproject.toml
└── README.md
```

## Training

The training code for the two 90 s checkpoints is in this project.

```bash
# Multi-GPU (DDP via torchrun). Point DATA_ROOT at the *_dataset_90s*.h5 directory.
DATA_ROOT=/path/to/input_h5_90sec torchrun --standalone --nproc_per_node=4 \
  scripts/train_rp90_motion.py --config configs/train_rp90_motion.json --num-workers 4
# Single-GPU:
DATA_ROOT=/path/to/input_h5_90sec \
  python scripts/train_rp90_motion.py --config configs/train_rp90_motion.json
```

- `scripts/train_rp90_motion.py` is the driver. It applies the DWA multi-task
  loss with AMP, DDP and DistributedSampler, and passes raw Z to the polarity
  head through `z_raw`. It builds `MTAN_R2UNet_RP90_Motion` from the config.
  Older `_xl` and architecture flags are accepted and ignored. `batch_size` in
  a config is per GPU, so the global batch is `batch_size` times the number of
  GPUs.
- `configs/train_rp90_motion.json` is the earlier **v45** config, with
  `nb_filters` `[6,12,18,24,32]`, a deeper polarity head, `softmax_ce`, and
  `pol_stream_width_mult=3` (331,336 parameters). The shipped `redpan_motion`
  (v49) records its own configuration in
  `redpan_motion/checkpoints/redpan_motion/config.json`,
  which the driver also accepts through `--config`. Set `data_root` or
  `DATA_ROOT` and the per-dataset `data` block.
- `configs/train_edge_rp90.json` trains EdgeRP90. The other files in
  `configs/` train larger or retuned variants. No checkpoint from those
  variants is shipped.
- `redpan_motion/data/dataset_v2.py` holds `MultiDatasetH5`, the loader for
  several datasets at once, with mosaic augmentation, a category whitelist and
  highpass per dataset, and category weights.
- `redpan_motion/training/losses.py` holds `MultiTaskLossWithPolarity` and DWA.

The training HDF5 layout is `waveforms (N,3,9000)`, `labels (N,9000,3)` as
one-hot P/S/N, and `masks (N,9000,2)` for the event mask. `MultiDatasetH5`
documents the full schema for each dataset.

## Tests

```bash
pytest tests/ -v
```

## Citation

If you use RED-PAN-Motion, please cite the RED-PAN paper:

```bibtex
@article{Liao2022,
  title = {RED-PAN: Real-Time Earthquake Detection and Phase-Picking With Multitask Attention Network},
  volume = {60},
  ISSN = {1558-0644},
  url = {http://dx.doi.org/10.1109/TGRS.2022.3205558},
  DOI = {10.1109/tgrs.2022.3205558},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  author = {Liao,  Wu-Yu and Lee,  En-Jui and Chen,  Da-Yi and Chen,  Po and Mu,  Dawei and Wu,  Yih-Min},
  year = {2022},
  pages = {1–11}
}
```

If you use the SeisBench integration, please also cite SeisBench.
`CITATION.cff` has the software citation.

## License

MIT © 2025 tso1257771. See [LICENSE](LICENSE). Built on the original
[RED-PAN](https://github.com/tso1257771/RED-PAN) architecture.

RED-PAN-Motion uses these open-source projects. None of their code is copied
into this repository.

| project | license | used for |
|---|---|---|
| [PyTorch](https://pytorch.org) | BSD 3-Clause | the models, training and inference |
| [NumPy](https://numpy.org) | BSD 3-Clause | arrays |
| [ObsPy](https://www.obspy.org) | LGPL-3.0 | waveform and station metadata I/O |
| [SeisBench](https://github.com/seisbench/seisbench) | GPL-3.0 | the optional `seisbench` extra, `redpan_motion.integrations.seisbench` |
| [TensorFlow](https://www.tensorflow.org) | Apache 2.0 | the optional `tf` extra, weight conversion only |
