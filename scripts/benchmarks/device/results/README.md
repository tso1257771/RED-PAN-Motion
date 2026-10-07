# Measurement runs

Each JSON holds the full configuration, machine state before/after, and **every
individual repeat** (`latency_all_ms`), so any summary here can be recomputed.

| tag | machine condition | use |
|---|---|---|
| `clean_*` | **Idle machine.** stdev 0.4-2.7 ms. | **The reference results.** Everything quoted in `../README.md` comes from these. |
| `sustained_*` | Idle machine, 180 s continuous load. | Thermal throttling and steady-state power. |
| `triton*` | Idle machine, served through Triton. `triton_rep1`/`rep2` are two independent repeats; `triton_clean` was a cold container and reads ~12 ms high. | Serving overhead. |
| `loaded_*` | Machine running a production Triton pipeline at ~150% CPU. | Kept for contrast: 1-thread rows stay usable, multi-thread rows show stdev up to 168 ms and are **not** valid scaling measurements. |
| `contended_*` | Same, and RED-PAN 60 s here still uses the **unfixed** ONNX export (9.2 s/window). | Historical record of the `repeat_interleave` export pathology. |
| `fixedcheck_*` | First run after the export fix: 9,229 ms -> 69.2 ms. | Evidence for the 133x fix. |

`onnx_manifest.json` records each exported graph's size, parameter count and
torch-vs-onnxruntime parity. `cmp_via_triton.log` compares the torch RED-PAN 60 s
export against a tf2onnx graph of the same model.

Machine for all runs: Jetson Nano Developer Kit, JetPack 4.6.1, 4x Cortex-A57
@ 1.479 GHz, 4 GB, MAXN, onnxruntime 1.14.1 CPU.
