# Benchmark harness for the RED-PAN-Motion manuscript

This folder holds the code that produces the evaluation numbers of the manuscript (Tables II–V,
Fig. 3, Section II-E, the streaming replay and the INT8 check) from the 90 s HDF5 archives and the
other test sets listed under [Data](#data). Every scorer was run on the archived outputs of the
manuscript runs and reproduces the published values exactly; every runner was checked against the
archived per-record outputs (see [Verification](#verification)).

## Setup

```bash
pip install -e ..                       # the redpan_motion package (repository root)
pip install -r requirements.txt         # pinned versions (Python 3.10-3.12)
cp config.example.yaml config.yaml      # then edit the data locations
export RPM_BENCH_CONFIG=$PWD/config.yaml
python -m pytest -q tests/              # 43 tests, synthetic data only, 1.5-2 min
```

Run every command from `benchmarks/`: the entry points import `rpm_bench` from there (no
`PYTHONPATH` is needed once the package is installed), and relative locations resolve against the
current directory.

Locations come from one place: the defaults in `rpm_bench/config.py` (data under `../data`, outputs
under `./results`, both relative to the current directory), overridden by a YAML file (`--config` on
any entry point, or `RPM_BENCH_CONFIG`) and then by environment variables `RPM_BENCH_<KEY>` (for
example `RPM_BENCH_H5_ROOT=/data/h5`). Unknown YAML keys are reported as a warning; a missing config
file, a `{key}` reference to an unknown key, or a reference cycle is an error. Every entry point has
`--help`. A runner that reads no records (a wrong location) exits with an error instead of writing an
empty output.

**Environments.** `requirements.txt` pins the versions the published numbers were produced and
re-verified with; the pins install on Python 3.10-3.12 only (numpy 1.26.4 has no wheels for later
versions). The INT8 check is documented with torch 2.5.1, onnx 1.21.0 and onnxruntime 1.24.2, as
pinned. The one-off conversion of the RED-PAN 2022 model ([below](#red-pan-2022-model)) ran in a
separate environment, TensorFlow 2.15.1 with PyTorch 2.10; only the converted checkpoint (md5 given
there) enters the benchmark. The synthetic tests also pass with Python 3.14, numpy 2.5 and pandas
3.0, but the published values were checked only with the pins.

Model keys: `edge` (Edge-RED-PAN-Motion, shipped checkpoint `edge_rp90`), `rpm` (RED-PAN-Motion,
shipped `redpan_motion`), `redpan` (RED-PAN, the 2022 paper model; see
[RED-PAN 2022 model](#red-pan-2022-model)), `redpan_240107` (shipped `redpan_60s`, used only for the
package's S-P table). Baselines: `phasenet_stead`, `phasenet_instance`, `eqt_stead`, `eqt_instance`.

## Commands per table and figure

`M` = each of `edge rpm redpan`. Runners write to `results/` (layout below); scorers read from it.

| Output | Runs (once per model / dataset) | Scorer |
|---|---|---|
| **Table II** (8 test sets, single trigger, pooled noise) and the detector-trigger count | `run_static.py --model M --dataset D` for D in `ceed_nc ceed_sc crew geonet instance romplus stead_h5 tw` (GeoNet: the 2013–2014 holdout earthquakes); `run_noise.py --model M --pool P` for P in `STEAD_test GeoNet_test INSTANCE_test RockNet_test TW_test` | `score_table2.py` |
| **Table III** (6 shared sets, threshold 0.3; GeoNet noise = the GeoNet test split) | `run_static.py --model M --dataset D` for D in `stead geonet crew tw instance romplus stead_noise_test` and `run_static_noise.py --model M --pool GeoNet`; `run_native.py --model B --dataset D` for B in the four baselines and D in `stead geonet geonet_noise_test crew tw instance romplus` (`--max-eq 30000` for `phasenet_stead`/`eqt_stead` on `stead` and `tw`, as published) | `score_table3.py` (`--geonet-noise holdout` for the first submission's GeoNet noise; `score_table3_geonet_test_noise.py` prints both) |
| Table III P and S pick F1 (no detection gate, all seven configurations) | the Table III runs, with `--picks` added to the `run_static.py` and `run_static_noise.py` runs of the RED-PAN models (`--evids` limits `stead` to the 16,301 held-out earthquakes) | `score_table3.py --pick-f1` |
| **Table IV** and the S-P threshold fit | the Table II noise runs; `run_static.py --model M --dataset D` for D in `ceed_nc ceed_sc crew geonet instance romplus stead_h5 tw` | `score_table4.py` (add `--models ... redpan_240107` for the package table) |
| **Fig. 3** (`f1_vs_sp_time.csv`) | the Table III static runs and the pooled-noise runs; for the baselines also `run_native.py --model eqt_stead/eqt_instance --dataset geonet_noise_test` and `rocknet_noise_test` | `score_fig3.py`, then `plot_fig3.py` (the manuscript's plotting script; needs matplotlib) plots the CSV |
| **Section II-E** (trigger duration vs S-P) | the Table IV static runs of `edge` | `score_duration.py` |
| **Table V** (CEED polarity) | `run_polarity.py --model edge` and `--model rpm` | `score_table5.py` |
| **Streaming replay** | `run_streaming.py --model M --mode eq` and `--mode noise` (`run_streaming_shard.py` splits a replay over several processes); `--model redpan --mode noise --noise-mask mask` for the RED-PAN noise replay with the mask | `score_streaming.py` (`--noise-subdir noise_mask` for the mask replay); `check_streaming_windows.py` counts short 60 s windows and scores with and without them |
| **INT8 check** | — | `int8_pick_f1.py --calib minmax` / `--calib percentile`, or `--int8-onnx GRAPH` to score an existing INT8 graph |
| STEAD test-noise archive (input) | `build_stead_noise_test.py` | — |
| GeoNet 2013–2014 holdout (input) | `build_geonet_holdout.py --kind events` and `--kind noise` | — |
| 90 s HDF5 archives (input) | `python -m builder --dataset D --category C` (`python -m builder --list` prints the plan; see `builder/README.md`) | — |

Large splits can be run in parallel chunks: `run_static.py ... --rows 0:144500 --out chunk0.csv` (and
so on) writes one chunk. `--rows` slices only the earthquake split; the noise records are read whole
by every chunk, so pass `--no-noise` to all chunks but one. Combine the chunks with pandas
(`pd.concat([pd.read_csv(f) for f in chunks]).to_csv(out, index=False)`), not `cat`, which repeats
the header line.

### Results layout

```
results/static/<model>/<dataset>.csv             one row per (record, mask trigger)
results/noise/<model>/noisefp[_triggers]_<POOL>.csv
results/native/<model>/<dataset>.csv             SeisBench baselines, one row per record
results/polarity/<model>_polarity_per_event.csv
results/streaming/<model>/{eq,noise}/*.csv       tab-separated, one file per event / noise record
results/streaming/<model>/noise_mask/*.csv       60 s models, --noise-mask mask (score_streaming.py --noise-subdir noise_mask)
results/int8/<tag>/                              graphs, per-record picks, scores
results/f1_vs_sp_time.csv, fig_f1_sp.{pdf,png}   score_fig3.py, plot_fig3.py
```

`run_streaming.py` writes each event file atomically (`<EVID>.csv.tmp`, then renamed), so a stopped
run can be restarted and skips the finished events.

## Data

| Config key | Contents | Used by | Source |
|---|---|---|---|
| `h5_root` | 90 s HDF5 archives, one directory per dataset: `<DS>/<DS>_dataset_90s_singleEQ.h5` and `..._noise.h5`, each with groups `/<category>/<split>/waveforms` (N, 3, 9000) float32, raw counts, E/N/Z, and a sidecar `..._metadata.csv` (`hdf5_index, sample_id, category, split, p_arrival_sample, s_arrival_sample, ...`). Directories: `CEED_NC CEED_SC CREW GeoNet INSTANCE ROMPLUS RockNet STEAD TW` | static runs, noise runs, INT8 | built from TW (restricted, Central Weather Administration), CEED (SeisBench), CREW (SeisBench), GeoNet, INSTANCE, ROMPLUS (RoSE), RockNet and STEAD with the archive builder in `builder/` (`python -m builder`; see `builder/README.md`) |
| `stead_noise_test_dir` | `STEAD_dataset_90s_noise_test.h5` (+ metadata), the 23,526 official STEAD test noise records as 90 s windows (md5 `0af03c69c2bde5159e980397dc01705d`) | `STEAD_test` pool, `stead_noise_test` | `build_stead_noise_test.py` from `stead_root` |
| `stead_root` | original STEAD release `merge.hdf5`, `merge.csv`, and the test trace list `test.npy` (126,566 names: 103,040 earthquakes, 23,526 noise; md5 `debefda348742abad1ce72f9d8541838`) | Table III, Fig. 3, noise builder | STEAD (Mousavi et al., 2019) |
| `geonet_holdout_root` | GeoNet 2013–2014 holdout: `metadata.csv` + `waveforms/<trace>.npy` (70,583 earthquakes, 270 s, P at 90 s) and `metadata_noise.csv` + `noise_waveforms/` (22,479 noise records, 120 s) | Table III, IV, Fig. 3 | `build_geonet_holdout.py` from `geonet_source_root` (the GeoNet benchmark dataset: `event_dataset/{metadata,waveform_data}`, `noise_dataset/{metadata,waveform_data}`) |
| `ceed_cache`, `ceed_test_ids` | SeisBench CEED cache (`metadata*.csv`, `waveforms<region><year>.hdf5`) and `ceed_test_ids_100k.npy` (99,998 NC + SC traces, 2021–2023; md5 `f96aef3fe2e28dc7e5f3b31bf2d196db`, 14 MB, distributed with the release, not in git) | Table V | CEED (SeisBench) |
| `eew_root` | `EQ_Mw4_to_6.9/<network>_<evid>/*.sac` (437 events) and `noise/sac_120s_{CWB_StrongMotion,Palert}/<id>/*.sac` (4,370 records of 120 s) | streaming replay | Taiwan strong-motion and P-alert records (restricted) |
| `redpan_paper_ckpt` | the RED-PAN 2022 paper model in PyTorch | `--model redpan` | [RED-PAN 2022 model](#red-pan-2022-model) |

Small lists shipped here: `data/eew_matched800_noise.txt` (the 800 noise records the original 60 s
RED-PAN replay was run on; `score_streaming.py` also reports scores on this subset) and `data/int8/`
(INT8 calibration and evaluation records).

## RED-PAN 2022 model

RED-PAN is evaluated with the model of the original paper (Liao et al., 2022), commit `93753bd` of
https://github.com/tso1257771/RED-PAN ("trained models described in original paper"), not with the
`REDPAN_60s_240107` release shipped in the package. Convert it once (TensorFlow is needed only here):

```bash
git clone https://github.com/tso1257771/RED-PAN && cd RED-PAN
git archive 93753bd pretrained_model/REDPAN_60s | tar -x -C /tmp/redpan_paper
git archive 2355054^ REDPAN_tools | tar -x -C /tmp/redpan_tools     # the TF model code of that era
cd <RED-PAN-Motion>
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python scripts/convert_redpan_60s.py \
    --tf-dir /tmp/redpan_paper/pretrained_model/REDPAN_60s --tf-tools /tmp/redpan_tools \
    --out <data_root>/checkpoints/redpan_60s_paper --verify
```

Expected: TF `train.hdf5` sha256 `0718a299d1ffcbed9632cbe5c46e86d806409c843bd5f6ac236964e60ca729d3`,
349,685 trainable parameters, picker / detector parity about 2e-7, and `best.pt` md5
`cc99ae8ff1e9142c0cefd1a21f66ab33` (TensorFlow 2.15.1, PyTorch 2.10).

## Runtimes

Measured on one RTX 2080 Ti with a Xeon W-2125 (8 threads); the runners are CPU-bound by
preprocessing, so several processes in parallel share one GPU well.

| Step | Approximate time |
|---|---|
| `run_static.py`, one model | 30–40 records/s per process: CREW 25 min, INSTANCE 20 min, GeoNet holdout 1 h, STEAD test 1.2 h, TW 3–4 h, CEED-SC (578k) 5 h (or 1.5 h in four `--rows` chunks) |
| `run_noise.py`, all five pools, one model | 15–25 min |
| `run_native.py`, one baseline, one dataset | 25 records/s (CREW 30 min, TW 3 h) |
| `run_polarity.py`, one model | about 25 min for 99,998 records |
| `run_streaming.py`, one model | earthquakes 45 min; noise 2.2 h (90 s models), 4.4 h (60 s RED-PAN) |
| `int8_pick_f1.py`, one variant | 3 min |
| scorers | 1–15 min each (Table IV and Fig. 3 are the slowest) |
| `tests/` (synthetic) | 1.5-2 min (43 tests) |

## Verification

Run on the benchmark PC before publication:

- **Scorers on the archived outputs** reproduce the published values exactly: Table II (24 cells with GeoNet on the 2013–2014
  holdout, macros, n, joint FP, detector-trigger counts 624 / 1,194 / 1,518), Table III (42 cells, CREW
  recall), Table IV and the four S-P tables (`edge_rp90`, `rp90_motion_v49`, `redpan_60s` for the paper
  model and for 240107), Fig. 3 (35 rows: F1, recall, n, thresholds), Section II-E (1,540,216
  triggers, r = 0.548, ρ = 0.670, bin medians), Table V (16 cells, McNemar counts and p), the streaming
  replay (best F1 0.9721 / 0.9850 / 0.9676; fire-ASAP delays 1.090 / 0.877 / 0.260 s on the full noise,
  0.864 / 0.737 / 0.260 s on the 800 subset) and the INT8 check (all variants of the results note).
- **Runners vs the archived per-record outputs:** on every record whose window needs no random
  padding, `run_static.py` and `run_noise.py` are bit-identical, or within 6e-7 (float32 rounding on
  GPU or CPU), with identical rows, triggers and picks (all three models), and `run_native.py` is
  bit-identical (PhaseNet, EQTransformer); `run_polarity.py` is bit-identical to the original script (both models, on a
  synthetic CEED cache, since the CEED cache was not on that PC); `run_streaming.py` matches the
  archived replay files (delays exact, probabilities to 4e-7); `build_stead_noise_test.py` reproduces
  the published file to 3e-5 counts. The packaged models load the shipped checkpoints, whose weights
  are identical tensor by tensor to the ones benchmarked.
- **Not bit-reproducible by design:** windows that extend past a record are filled with
  spectrum-matched noise from an unseeded generator (as in the manuscript runs), so a re-run differs
  slightly on those records; scores agree to the reported precision. `--seed` (`run_static.py`,
  `build_geonet_holdout.py`) makes a re-run repeatable, but it does not reproduce the archived
  padding.
- **GeoNet-holdout noise records:** `iter_geonet_noise` (`restore_taper`) replaces the tapered record
  edges with spectrum-matched noise from an unseeded generator, so runs that read the whole 120 s
  record (`run_native.py --dataset geonet`) differ between re-runs on those records; the 90 s static
  crop leaves the edges out.
- **Re-verified after the code cleanup (cce69a9):** every scorer's output is byte-identical to the
  pre-cleanup code (9314d78) on the archived outputs, and the runners are bit-identical between the
  two versions.

## Known caveats (kept so that the published values reproduce)

- **Streaming replay, 60 s RED-PAN noise:** the original noise replay stored the P probability in the
  `Mask_probability` column (the mask was computed but not written), so RED-PAN's noise false alarms
  are gated on P probability where the other models are gated on the mask. `run_streaming.py` keeps
  this by default (`--noise-mask legacy`); `--noise-mask mask` writes the mask. Gated on the mask,
  RED-PAN's best F1 on the full noise is 0.9655 (0.9676 published) and the fire-ASAP delay stays
  0.260 s, at precision 0.9464 (0.9433). The 90 s replays always stored the mask.
- **GeoNet holdout noise (Table III):** the 22,479 noise records come from a seeded split of the GeoNet
  noise that is independent of the 90 s archive's split; 13,752 of them are in the archive's training
  split and 4,423 in its validation split. The holdout earthquakes are not in the archive. Tables II,
  IV and Fig. 3 use the pooled test noise instead, and `score_table3_geonet_test_noise.py` scores the
  Table III GeoNet column on the GeoNet test noise (16,384 records, all from 2024), as
  `score_table3.py` does by default.
- **Pick F1 (Table III, `--pick-f1`):** every model's scored pick follows the native SeisBench rule
  (`rpm_bench/picks.py`): the highest `find_peaks` peak within the tolerance around the label, or the
  highest peak of the record on noise. The first submission's P / S F1 used the pick of the
  highest-mean trigger row instead, which for the RED-PAN models is the argmax inside that mask
  trigger; a RED-PAN noise record without a trigger could not be a false positive. For the
  baselines the two rules agree.
- **Static windows** place the labeled P at 10% of the model window (an evaluation convenience that uses
  the label, the same for all RED-PAN models).
- **INT8:** the accuracy change depends on the calibration method (MinMax on real records about −2.6 pp
  P F1, Percentile about −0.8 pp, the archived synthetic-calibrated graph about 0); name it when citing.
- **Trigger end sample:** obspy's `trigger_onset` returns the last active sample as the trigger end
  (inclusive). `extract_triggers` (static runs and the noise trigger rows), the trigger statistics
  of `run_polarity.py` and the peak-in-trigger test of the streaming replay use `[lo:hi]` or
  `lo <= i < hi`, which leave that sample out, while the per-record detector-trigger maximum of
  `run_noise.py` uses `[lo:hi + 1]`. The effect is at most one sample per trigger.
- **60 s replay padding (vendored RED-PAN code):** `sac_len_complement` appends `len(data)` zeros per
  missing sample, so a trace k samples short grows to 2^k times its length (the zeros lie after the
  record end; a warning is logged when k > 1). `stream_standardize` pads a short window with zeros
  inserted before its last sample, so a window that starts before the record start keeps its data at
  the window start, shifted in time; `run_streaming.py` logs how many windows of each event were short.
  In the manuscript's earthquake replay no window is short: the labeled P is 65 s into every record,
  so the earliest window starts 4 s inside it (`check_streaming_windows.py` counts them).
- **One-sample window offset (streaming replay):** each window is sliced including both ends and cut
  to its first 6,000 (60 s) or 9,000 (90 s) samples, so its newest sample is dropped. The model sees
  data up to 0.01 s before the time the delay is measured from: every delay is 0.01 s longer than the
  data used, the same for every model.
- **`polarity_at_P` column of the static CSVs:** channel 0 of the polarity output at the P pick, which
  for the 3-class [N, U, D] head of `edge` and `rpm` is P(None), not a signed polarity. No scorer
  reads it (Table V comes from `run_polarity.py`); the name is kept so that the CSVs match the
  archived outputs.
- **Noise-run input differs from the static runs:** `run_noise.py` filters the record without the
  demean and 5% Tukey taper of `run_static.py`, and z-scores with `std` where `std > 1e-8` (else 1)
  instead of `std + 1e-10`, as the manuscript's noise run did.

## Not included

Left out because no manuscript number depends on them: the sliding-window mode and its padding, the
TensorFlow RED-PAN wrapper and the TF-era noise script (the published RED-PAN rows use the PyTorch port
of the 2022 model), the earlier Table II scorer (it counted noise records as missed earthquakes) and the
step-by-step re-run decomposition, the legacy summaries and sweeps (`summarize_3way_*`, `analyze_*`,
`joint_threshold_sweep_*`), wrappers and benchmarks of other models (PhaseNet+, PhaseNO, earlier
RED-PAN-Motion versions), the CEED pick/detection and band-pass polarity variants, the moving-window
normalization path, and the on-device latency/energy scripts (already in `scripts/benchmarks/device/`).

The 90 s archives are an input here. `builder/` rebuilds them from the dataset releases
(`python -m builder`; see `builder/README.md` for each dataset, including the CEED converter). The TW
waveforms are restricted, so the TW archives cannot be rebuilt from public data.
