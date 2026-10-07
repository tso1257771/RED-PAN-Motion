# Rebuilding the 90 s archives

The static, noise and INT8 runners read the 90 s HDF5 archives under `h5_root`
(`<DATASET>/<DATASET>_dataset_90s_{singleEQ,noise}.h5` plus `_metadata.csv`). This package rebuilds
them from each dataset's original release. It is RED-PAN's `redpan/data/builder`, ported with the
changes listed at the end.

Each (dataset, category) has an adapter. The adapter reads the release, windows every trace to 90 s
(9,000 samples at 100 Hz, channels E, N, Z, raw counts), and pads with spectrum-matched noise when the
trace is shorter. It then assigns train, val or test, and the writer stores
`/<category>/<split>/waveforms` (N, 3, 9000) float32 with one metadata row per sample.

Two inputs of the benchmark are built by their own scripts:
- the official STEAD test noise (`build_stead_noise_test.py`, 23,526 records, key `stead_noise_test_dir`);
- the GeoNet 2013–2014 holdout (`build_geonet_holdout.py`, key `geonet_holdout_root`).

## Running it

Run from `benchmarks/`. Locations come from the same config as the rest of the harness: `--config`,
`RPM_BENCH_CONFIG`, or the environment variables `RPM_BENCH_<KEY>`. The builder adds the source keys
below to the harness keys. Outputs go to `<build_root>/<DATASET>/`, a separate directory from
`h5_root`, so a rebuild never overwrites the archive the benchmark reads. To benchmark on a rebuilt
archive, point `h5_root` at `build_root`.

```bash
python -m builder --help
python -m builder --list                                         # plan, resolved sources, missing files
python -m builder --dataset STEAD --category noise --validate    # one archive
python -m builder --dataset CREW --category singleEQ --max-samples 200 --overwrite   # smoke run
```

To rebuild everything, run every pair that `--list` prints, each with `--validate`.
The pairs are independent, so they can run in parallel.

| Key | Default | Contents |
|---|---|---|
| `build_root` | `{data_root}/input_h5_90sec_rebuilt` | output archives |
| `stead_root` | `{data_root}/STEAD` | `merge.hdf5`, `merge.csv`, `test.npy` (shared with the harness) |
| `instance_root` | `{data_root}/INSTANCE` | `Instance_events_gm.hdf5`, `metadata_Instance_events_v2.csv`, `Instance_noise.hdf5`, `metadata_Instance_noise.csv` |
| `geonet_source_root` | `{data_root}/GeoNet_benchmark` | `event_dataset/{waveform_data/units,metadata}`, `noise_dataset/{waveform_data/units/waveforms_units_noise.h5,metadata/metadata_noise.csv}` (shared with the harness) |
| `crew_root` | `{data_root}/seisbench_cache/datasets/crew` | SeisBench CREW download: `chunks`, `metadata{NNN}.csv`, `waveforms{NNN}.hdf5` |
| `obst_root` | `{data_root}/OBSTransformer` | `training_data.hdf5` |
| `rocknet_root` | `{data_root}/RockNet` | `Luhu_hdf5/Luhu_dataset.h5`, `metadata/partition/*_partition.npy` |
| `romplus_root` | `{data_root}/ROMPLUS` | `sac/{year}/{event}/*.sac`, `romplus_metadata.csv` |
| `ceed_nc_h5_dir` | `{data_root}/CEED_NC_redpan_h5/CEED_NC` | CEED Northern California as 90 s windows: `CEED_NC_dataset_90s_singleEQ.h5` + `_metadata.csv` (see CEED below) |
| `ceed_sc_h5_dir` | `{data_root}/CEED_SC_redpan_h5/CEED_SC` | CEED Southern California as 90 s windows: `CEED_SC_singleEQ_s???.h5` shards + `_metadata.csv` / `_metadata_polarity.csv` |
| `tw_eq_root` | `{data_root}/TW_eq_data` | **restricted**: `TW_2012_2019_180s/`, `metadata_TW_2012_2019_180s/`, `first_motion/available_data.csv` |
| `tw_noise_root` | `{data_root}/TW_noise_data` | **restricted**: `metadata/TW_noise/` (hourly SAC), `metadata/pred_TW_noise/` |

The other entry points of the harness log a warning for these extra keys if they are in a shared
YAML file, and otherwise ignore them.

## Per dataset

Splits are 70 / 15 / 15 % unless stated otherwise. Event-bearing archives split by a salted hash of
the event key, so all stations of one earthquake share a split. Noise archives draw an independent
per-record split from a generator seeded with `--seed 42`. The archives in the benchmark are
`singleEQ` and `noise`. `Ponly` and `Sonly`, where available, are training augmentation.

| Dataset | Archives | Source | Split | Notes |
|---|---|---|---|---|
| STEAD | `singleEQ`, `noise` | STEAD release (Mousavi et al., 2019) | singleEQ: hash of `source_id`; noise: 80 / 20 train / val | Noise leaves out the 23,526 noise traces of `test.npy` (`--exclude-traces`, default `<stead_root>/test.npy`); that set is the benchmark's STEAD test noise. Table III scores STEAD on the 16,301 `test.npy` earthquakes that are in neither the train nor the val split of `singleEQ`. |
| INSTANCE | `singleEQ`, `noise` | INSTANCE (Michelini et al., 2021), events (ground motion) and noise | singleEQ: hash of `source_id`; noise: per record | |
| GeoNet | `singleEQ`, `noise` | GeoNet benchmark dataset | by year: 2015–2021 train, 2022–2023 val, 2024 test; other years are left out | The event archive has no test split (no 2024 events in the release). The 16,384 test-split noise records are the GeoNet pool of the pooled noise. The 2013–2014 holdout comes from `build_geonet_holdout.py`. |
| CREW | `singleEQ`, `Ponly`, `Sonly` | SeisBench CREW (`seisbench.data.CREW()`) | hash of the event | `singleEQ` uses the Pn and Sn picks only (`--crew-phase-pair mantle`, the default here). CREW is not used in training. |
| OBSTransformer | `singleEQ` | OBSTransformer `training_data.hdf5` | hash of the event | |
| RockNet | `noise` | RockNet Luhu dataset | the published `*_partition.npy` files; seeded per-record fallback for events not in them | |
| ROMPLUS | `singleEQ`, `Ponly`, `Sonly` | ROMPLUS SAC files and catalog | hash of `{year}_{event}` | |
| CEED_NC | `singleEQ` | CEED (SeisBench), via a 90 s intermediate | the intermediate's split, kept as is | `--with-polarity` (default on) carries U / D / N first motions. |
| CEED_SC | `singleEQ` | CEED (SeisBench), via 90 s shards | the shard's split, kept as is | As CEED_NC. |
| TW | `singleEQ`, `Ponly`, `Sonly`, `noise` | Central Weather Administration waveforms and catalog | events: hash of the event; noise: hash of the hour | **Restricted.** The waveforms are not public and are not distributed with this repository. Access goes through the Central Weather Administration (Taiwan). Without them the TW archives cannot be rebuilt, and the TW results cannot be reproduced from the release. |

**CEED.** The CEED adapters repack 90 s intermediate shards made from the SeisBench CEED cache
(`ceed_cache`). For CEED_SC the chain is:
1. `python -m builder.ceed_convert --region sc` writes `CEED_SC_singleEQ_sNNN.h5` + `_metadata.csv`
   to `ceed_sc_h5_dir`, with CEED's own train / dev / test split. This is `prepare_ceed_h5_direct.py`,
   recovered from an internal 2026-06-03 snapshot of the authors' working directory. With
   `--legacy-shard-index` and the same seed, the port is byte-identical to the original on synthetic
   data.
2. In-place clean-up of the shard metadata by three RED-PAN scripts. They are not in a public commit,
   because RED-PAN's `scripts/` directory is git-ignored, and they are not ported here:
   - `filter_ceed_suspicious.py` drops traces whose phase list is not strictly P, S, P, S, ...;
   - `clean_corrupted_h5.py` drops rows with corrupted pick counts;
   - `add_polarity_to_metadata.py` adds `p_polarity` from the CEED catalog.

   In the CEED_SC intermediate this removed 161,533 of 1,840,090 rows (8.8%). The per-shard
   `*_metadata_polarity.csv` files that `--with-polarity` prefers were written by a script that was
   not identified.
3. `python -m builder --dataset CEED_SC --category singleEQ`.

**CEED_SC shard overwrites.** The original converter numbered shards per split but named them
without the split, so a later split's shard overwrote an earlier split's shard with the same number.

| Split | Converter wrote | Survived in the intermediate | Lost |
|---|---|---|---|
| train | 1,902,252 | 1,202,252 (shards 14–38, 2015–2019) | shards 0–13: 700,000, 1999–2015 |
| val | 486,231 | 36,231 (2020) | shards 0–8: 450,000 |
| test | 651,607 | 601,607 (2021–2023) | shard 9: 50,000 |

H5 files and CSVs were overwritten together, so every surviving row is consistent. The CEED-SC test
set of Tables II and IV, 577,994 records after the clean-up, is part of CEED's 2021–2023 test split,
so it remains held out. The CEED_SC singleEQ training split covers 2015–2019 only. The CEED_SC
MOSAIC archive, built separately from the SeisBench cache by RED-PAN's mosaic adapter, does not
go through this converter.
`builder.ceed_convert` numbers shards across splits, so nothing is overwritten;
`--legacy-shard-index` restores the original numbering.

**CEED_NC.** The archive's NC intermediate is a single merged file,
`CEED_NC_dataset_90s_singleEQ.h5` + `_metadata.csv` (1,032,665 records), which the CEED_NC adapter
reads. The step that made it was not found. `builder.ceed_convert --region nc` writes shards in the
CEED_SC layout instead, which the CEED_NC adapter does not read.

## Agreement with the benchmark archive

Small builds (`--max-samples 300`) were compared, by sample id, with the archive the benchmark
read (`input_h5_90sec_v3`).

| Archive | Sample ids, splits, P / S samples, polarity | Waveforms |
|---|---|---|
| GeoNet `noise` | identical (300 / 300) | identical (100 / 100 compared) |
| CEED_SC `singleEQ` | identical | identical |
| STEAD `singleEQ` | identical | real samples identical; the spectrum-matched padding differs (unseeded generator, as in the original runs) |
| STEAD `noise` | none of `test.npy`; split differs | see below |

The STEAD noise archive in `input_h5_90sec_v3` predates the builder (February 2026). Its membership
is exactly all STEAD noise minus the 23,526 `test.npy` traces, 211,900 records, split 169,520 /
42,380 train / val. The builder reproduces that membership and the 80 / 20 proportions, but not the
per-record split. In 190 of 300 compared records the real 60 s also differs from the legacy file in
the samples, at the same amplitude scale. No benchmark reads this archive; it is training data only.

## Changes from RED-PAN's builder

- The spectrum-matched padding helpers are imported from `redpan_motion.utils.waveform`. They are
  identical to RED-PAN's `redpan/utils.py` versions (same syntax tree).
- `STEADNoiseAdapter` takes `exclude_traces`. The CLI passes `test.npy`, with an 80 / 20 split.
- `validators.assert_h5_consistency` accepts the polarity `N`, as the schema does. In RED-PAN's
  version, `--validate` rejected every CEED build with polarity.
- New command line: `python -m builder`, defaults from the harness config, and `--list`.
- Added `builder.ceed_convert` (the CEED converter above), with shard numbering across splits.
- Left out, because no benchmark reads them:
  - the MOSAIC / EEWA / MMWA augmentation adapters, which depend on RED-PAN's training code;
  - the deprecated `singleEQ_zeropad` category;
  - the legacy-parity helper (`parity.py`).
