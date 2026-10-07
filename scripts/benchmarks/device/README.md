# On-device efficiency benchmark (edge hardware)

Measures latency, energy and peak RAM of the RED-PAN models on real edge
hardware, standalone and served through NVIDIA Triton.

Reference run: **NVIDIA Jetson Nano Developer Kit**, JetPack 4.6.1 (L4T R32.6.1),
4x Cortex-A57 @ 1.479 GHz, 4 GB shared RAM, MAXN power mode, onnxruntime 1.14.1
(CPU execution provider), 60 repeats per configuration, idle machine.

---

## 1. Reading the numbers: what every term means

If a column name is unclear, it is defined here. Nothing below assumes prior
familiarity with deployment jargon.

### Timing columns

| term | meaning |
|---|---|
| **window** | One input chunk the model consumes at once: 90 s = 9,000 samples at 100 Hz (60 s = 6,000 for RED-PAN 60 s). Every latency figure is "time to process one window". |
| **median** | The middle value of N repeats. Used instead of the mean because a single scheduling hiccup should not move the headline number. |
| **min** | Fastest repeat: the machine's best case, least disturbed by other processes. |
| **p90** | 90th percentile. 9 out of 10 windows completed at least this fast. Describes the slow tail, which matters for real-time deadlines. |
| **sd** (stdev) | Spread across repeats. **This is the honesty check**: a few ms means the measurement is trustworthy; tens of ms means something else was competing for the CPU and the number should not be published. |
| **ms/1ks** | Milliseconds per 1,000 input samples. Necessary because the 60 s model sees 6,000 samples per window and the 90 s models 9,000, so their per-window times are not comparable. This normalizes them. It is the measured counterpart of the paper's `kMAC/sample`. |

### Throughput, CPU and memory

| term | meaning |
|---|---|
| **thr** (threads) | How many CPU cores onnxruntime may use for a single window (`intra_op_num_threads`). `thr 1` = one core, the fairest cross-model comparison; more threads = same work split across cores. |
| **RTF** (real-time factor) | How much faster than real time the model runs. **RTF 767x** means 90 s of recording is processed in 0.117 s, so one second of computing covers 767 seconds of recording. Practically: one CPU core could keep up with roughly 767 stations streaming continuously (ignoring window overlap and I/O). |
| **eff** (core efficiency) | CPU time / (wall time x threads). **1.0** = every core granted was busy the whole time (perfect scaling). **0.5** = half the granted CPU was wasted, through poor parallel scaling or competition with other processes. *Caveat: our background sampler thread contributes CPU time, so `eff` reads ~1.2 at `thr 1`. Wall-clock latency is unaffected.* |
| **RSS** (Resident Set Size) | The physical RAM a process actually occupies right now, in MB. Not disk, not reserved-but-unused address space. **This is the number that decides whether a model fits on a small device.** `peakRSS` is the maximum seen during a run. |
| **baseline RSS** | RAM used by bare Python + numpy + onnxruntime before any model loads. Subtract it from peak to get the model's own cost. |

### Energy and power

| term | meaning |
|---|---|
| **mW / mJ** | Milliwatts (power, rate of energy use) and millijoules (energy, power x time). 1 J = 1 watt sustained for 1 second, so 160 mJ = 0.16 J. |
| **VDD_IN** | Total power drawn by the whole board (CPU + GPU + RAM + I/O), read from the Jetson's on-board INA3221 sensor. |
| **VDD_CPU** | The CPU rail alone. |
| **idle baseline** | Board power with the benchmark not running (1,879 mW here, which includes the desktop session). |
| **E tot vs E dyn** | *Total* energy = board power x latency, so it includes the cost of the board merely being switched on. *Dynamic* energy = (board power - idle) x latency: the energy actually attributable to running the model. **E dyn is the number to compare models with.** |
| **throttling / drift** | Chips reduce their clock when hot. "Drift" is the latency change between the first and last quarter of a sustained run: ~0% means no throttling. |

### Serving (Triton) columns

| term | meaning |
|---|---|
| **Triton** | NVIDIA's inference server: it loads models and serves them over HTTP/gRPC, which is how a model is typically deployed rather than called in-process. |
| **model repository** | The directory of models Triton serves; one subdirectory per model. |
| **config.pbtxt** | A model's configuration file: input/output names, shapes, and placement. |
| **instance_group / KIND_CPU** | How many copies of the model run and on which device. `KIND_CPU, count 1` = a single CPU-resident copy. |
| **max_batch_size** | How many windows may be combined into one call. `0` means the shape is fixed and batching is off. |
| **client median** | Latency as the *caller* experiences it: request encoding, network, queueing, inference and response decoding. |
| **srv infer** (`compute_infer`) | Time Triton reports for running the model itself, excluding transport. |
| **queue** | How long a request waited for a free model instance. |
| **in/out** (`compute_input`/`compute_output`) | Time spent converting and copying tensors into and out of the model. |
| **overhead** | `client median - srv infer`: everything serving adds on top of raw inference. |
| **req/s** | Completed requests per second at concurrency 1. |

### Model and graph terms

| term | meaning |
|---|---|
| **ONNX** | A portable file format for a trained network, so the same model can run under onnxruntime, Triton, and other engines without the original framework. |
| **opset** | The version of the ONNX operator set a file targets. |
| **node / op** | One operation in the graph (`Conv`, `Add`, `Transpose`, ...). A graph is a few hundred to a few thousand nodes. |
| **MAC / GMAC / kMAC** | Multiply-accumulate: one multiply plus one add, the unit of convolution arithmetic. G = billions, k = thousands. **A MAC count is a static prediction of cost, not a measurement** - this benchmark exists precisely because the two differ. |
| **fp32 / INT8** | Numeric precision of weights and activations: 32-bit float (normal) vs 8-bit integer (compressed ~4x in principle). |
| **quantization / PTQ / QDQ / calibration** | Converting a float model to integers. *PTQ* = post-training quantization, done after training with no retraining. *QDQ* = the quantize-dequantize graph style used. *Calibration* = running sample inputs to learn each tensor's numeric range. |
| **`asimddp` / SDOT** | The ARM CPU instruction that makes INT8 arithmetic fast. **The Cortex-A57 in this Nano is ARMv8.0 and lacks it**, which is why INT8 is *slower* here, not faster. |
| **parity / max abs diff** | The largest absolute difference between two versions' outputs on identical input. `~1e-7` = identical to float32 rounding noise; `0.00e+00` = bit-identical; `~1e-3` on probabilities = same weights but differently optimized graphs. |
| **`Loop` / `SequenceInsert`** | ONNX control-flow operators. They appear when a Python loop is traced instead of becoming a vectorized op, and they are very slow - see the RED-PAN 60 s finding below. |
| **warmup** | Discarded first runs, before which timings reflect memory allocation and thread-pool startup rather than steady-state cost. |

---

## 2. Results (Jetson Nano, idle machine, 60 reps)

Standalone onnxruntime, CPU:

| model | window | 1 thread | best (3 thr) | ms/1000 samp | RTF | peak resident memory above runtime | E dyn |
|---|---|---:|---:|---:|---:|---:|---:|
| RED-PAN 60 s | 60 s | 63.6 ms | 38.9 ms | 10.60 | 943x | 30.3 MB | 72 mJ |
| **Edge-RPM fp32** | 90 s | **117.3 ms** | **64.8 ms** | 13.03 | 767x | 33.1 MB | 160 mJ |
| Edge-RPM INT8 | 90 s | 134.9 ms | 83.9 ms | 14.99 | 667x | 37.9 MB | 158 mJ |
| RP-Motion | 90 s | 329.3 ms | 140.3 ms | 36.59 | 273x | 23.4 MB | 406 mJ |

Standard deviation across repeats was 0.4-2.7 ms, idle baseline 1,879 mW. The memory
column is the process peak less the ~38 MB bare runtime (Python, numpy, onnxruntime),
that is, peak resident memory above the runtime.

1. **INT8 is 15% slower than fp32 at equal energy** (134.9 vs 117.3 ms; 158 vs 160 mJ).
   This CPU has no `asimddp`, the condition Section V-D names. On such hardware INT8
   buys file size only - and its *process* RSS is 4.8 MB higher, because the QDQ graph
   adds nodes and fp32 scale tensors. (The paper's 0.86 -> 0.22 MB refers to the largest
   single activation, which is not the same quantity as deployed memory.)
2. **A 4.47x MAC reduction delivers 2.81x latency and 2.5x energy** (329.3 -> 117.3 ms,
   406 -> 160 mJ): about 63% of the MAC-implied gain, and 2.17x at 3 threads.
3. **The kMAC/sample ordering survives but compresses.** Edge-RPM costs 1.23x RED-PAN 60 s
   per sample where the MAC counts predict 1.66x.
4. **No thermal throttling.** Over 180 s sustained: clocks pinned at 1,479 MHz, 26 -> 30.5 C,
   latency drift -0.6% (1 thread) and -1.0% (4 threads). Sustained throughput 8.45 windows/s
   at 1 thread, 13.68 at 4. Best scaling is at **3** threads; 4 regresses as the last core
   contends with the operating system.

Served through Triton (24.11-py3-igpu, CPU, one instance, two independent runs):

| model | client median | srv infer | overhead |
|---|---:|---:|---:|
| `edge_rp90` | 117.2 / 117.9 ms | 115.1 / 114.9 ms | 2.1-2.9 ms |
| `edge_rp90_int8` | 132.6 / 132.9 ms | 129.3 / 130.0 ms | 2.9-3.3 ms |
| `onnx_RP60` (tf2onnx) | 143.1 / 141.9 ms | 140.3 / 139.6 ms | 2.4-2.7 ms |

Served outputs match local onnxruntime to 1.31e-06 (fp32) and 0.00e+00 (INT8).
**Serving costs 2-3 ms, under 3%**: the served model is as fast as the in-process one.

### Two export findings

- **`repeat_interleave` must not be traced.** `redpan_60s.py` used
  `x.repeat_interleave(k, dim=2)` for Keras `UpSampling1D`. torch traces it into 8
  dynamic `Loop`/`SequenceInsert` regions that consumed 88% of graph runtime:
  **9,229 ms per window, versus 63.6 ms** after rewriting it as `unsqueeze -> expand ->
  reshape` (`_nearest_upsample`). Bit-identical in torch, 1.19e-07 against the original.
  A 133x difference that no MAC count predicts.
- **tf2onnx exports pay a layout tax.** A tf2onnx graph of RED-PAN 60 s carries 138
  `Transpose` + 105 `Squeeze` + 104 `Unsqueeze` around its 105 `Conv` nodes, because TF's
  channels-last Conv1D becomes unsqueeze -> transpose -> conv -> transpose -> squeeze.
  Served side by side it runs at 140 ms against 63.6 ms for the torch export of the same
  model (outputs agreeing to 3.3e-04).

---

## 3. Reproducing

Requires `onnxruntime`, `numpy`, `onnx`; export additionally needs `torch`.

```bash
cd scripts/benchmarks/device

python export_onnx_all.py          # shipped checkpoints -> ONNX (+ parity check)
python quantize_int8.py            # INT8 PTQ of the edge model

python bench_models.py --only edge_rp90_t1 --reps 60 --tag clean --header
bash   run_sweep.sh clean 60 "1 2 3 4"        # every model x thread count
python sustained.py --model edge_rp90 --threads 4 --seconds 180   # throttling
python profile_graph.py onnx/redpan_60s.onnx 6000                 # per-op profile
```

Paths are overridable with `RPM_ONNX_DIR`, `RPM_RESULTS_DIR`, `RPM_CKPT_DIR`;
results are written as JSON (including every individual repeat) to `results/`.

To measure **energy**, the board's power rails must be readable - they are root-only
by default, and this L4T build's `tegrastats` will not print power without them:

```bash
sudo chmod a+r /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_{power,voltage,current}*_input
```

Without this the energy columns are simply omitted; everything else still runs.

### Serving benchmark

```bash
docker run -d --name rp_triton_edgetest -p 7182:7182 -p 7183:7183 -p 7184:7184 \
  -v "$PWD/triton_repo:/models:ro" nvcr.io/nvidia/tritonserver:24.11-py3-igpu \
  tritonserver --model-repository=/models \
    --http-port=7183 --grpc-port=7182 --metrics-port=7184 \
    --model-control-mode=explicit --load-model=edge_rp90 --load-model=edge_rp90_int8

python triton_bench.py --reps 40 --tag triton        # RPM_TRITON_HOST / _PORT to retarget
python compare_graphs.py                             # torch export vs a served reference graph
```

---

## 4. Caveats

- ONNX files are **not** committed (they are regenerable in one command); `results/`
  holds the measurements, including per-repeat timings and machine state.
- The INT8 graph here is calibrated on 64 synthetic windows. That is valid for
  **latency**, but **not** for accuracy claims - the paper's INT8 used 192 held-out
  real traces.
- `eff` above 1.0 at 1 thread is the sampler thread's CPU time; wall latency is unaffected.
- The idle power baseline includes the desktop session, so dynamic energy is a
  conservative (slightly low) attribution.
- Latency is reported for a **single window at concurrency 1**. Deployment with
  overlapping sliding windows processes more windows per second of recording, so
  divide RTF by the overlap factor for a station-count estimate.
- On the reference board, `python` is 3.7 and `torch` caps at 1.13, while the package
  needs 3.8+ (`math.prod`) and torch >= 2.3 (`torch.amp.GradScaler`). The export scripts
  therefore import the model modules directly, bypassing `redpan_motion/__init__.py`.
  On a normal machine that shim is inert.
