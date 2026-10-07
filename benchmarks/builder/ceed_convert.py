"""Convert the SeisBench CEED release to the 90 s singleEQ shards the CEED adapters repack.

Ported from ``prepare_ceed_h5_direct.py`` (RED-PAN-motion ``scripts/``, the converter that wrote
``CEED_SC_redpan_h5``; its log line is "CEED SC -> RED-PAN H5 (direct, no TFRecords)"). For every
CEED trace with a P and an S pick (S at least 0.5 s after P and inside the 120 s record) it pads
both ends with spectrum-matched noise, cuts 90 s (9,000 samples) with the P at a random position
(10 s to 70 s, earlier for long S-P), keeps every P and S pick that falls in the window and assigns
CEED's own split (train / dev -> val / test). Output per region (``--region nc`` or ``sc``):
``CEED_<R>_singleEQ_sNNN.h5`` (``/singleEQ/<split>/waveforms`` (N, 3, 9000), one split per shard,
``--shard-size`` samples each) plus ``CEED_<R>_singleEQ_sNNN_metadata.csv``.

Locations come from the harness config (``--config`` / ``RPM_BENCH_CONFIG`` / ``RPM_BENCH_<KEY>``):
the SeisBench CEED cache ``ceed_cache`` (``metadata<r><year>.csv``, ``waveforms<r><year>.hdf5``) and
the output ``ceed_nc_h5_dir`` / ``ceed_sc_h5_dir``. See ``builder/README.md`` (CEED) for the steps
between this converter and the archive.

Shard names: the original converter counted shards per split but named them without the split, so
a later split's shard overwrote an earlier split's shard with the same number. In the CEED_SC
intermediate this kept train shards 14-38 (2015-2019; 1999-2015 overwritten), only the last, partial
val shard (36,231 of 486,231 val samples) and 12 of the 13 full test shards plus the partial one. Shards are now numbered across splits, so nothing is overwritten; ``--legacy-shard-index``
restores the original numbering (and its overwrites) to reproduce that intermediate's membership.
The P position and the padding come from an unseeded generator, as in the original run; ``--seed``
makes a re-run repeatable but does not reproduce the archived waveforms.

Example (run from ``benchmarks/``):
    python -m builder.ceed_convert --region sc
"""

from __future__ import annotations

import argparse
import ast
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from redpan_motion.utils.waveform import find_reference_signal, generate_matching_noise

from . import sources

log = logging.getLogger(__name__)

ORIGINAL_NPTS = 12000
DATA_NPTS = 9000
DT = 0.01
MIN_PS_RESIDUAL = 50
MIN_P_POSITION = 1000

REGION_CONFIG = {
    "nc": {"years": [str(y) for y in range(1987, 2024)], "prefix": "nc"},
    "sc": {
        "years": [
            "1999", "2000", "2001", "2002", "2003", "2004", "2005",
            "2006", "2007", "2008", "2009", "2010", "2011", "2012",
            "2013", "2014", "2015", "2016", "2017", "2018",
            "2019_0", "2019_1", "2019_2", "2020_0", "2020_1",
            "2021", "2022", "2023",
        ],
        "prefix": "sc",
    },
}  # fmt: skip

CEED_SPLIT_MAP = {"train": "train", "dev": "val", "test": "test"}


# ---------------------------------------------------------------------------
# Phase parsing
# ---------------------------------------------------------------------------
def _parse_numpy_str_list(s):
    s = re.sub(r"'\s+'", "', '", s.strip())
    return ast.literal_eval(s)


def _parse_numpy_int_list(s):
    return ast.literal_eval(re.sub(r"\s+", ", ", s.strip()))


def parse_all_ps_samples(row):
    """All P and S arrival samples of a CEED trace (every event in it)."""
    types_raw = row.get("trace_phase_type_list", "")
    arrivals_raw = row.get("trace_phase_arrival_sample_list", "")
    if pd.isna(types_raw) or pd.isna(arrivals_raw):
        p = row.get("trace_p_arrival_sample", np.nan)
        s = row.get("trace_s_arrival_sample", np.nan)
        return (
            np.array([int(p)]) if pd.notna(p) else np.array([], dtype=int),
            np.array([int(s)]) if pd.notna(s) else np.array([], dtype=int),
        )
    try:
        types = _parse_numpy_str_list(str(types_raw))
        arrivals = _parse_numpy_int_list(str(arrivals_raw))
    except (ValueError, SyntaxError):
        p = row.get("trace_p_arrival_sample", np.nan)
        s = row.get("trace_s_arrival_sample", np.nan)
        return (
            np.array([int(p)]) if pd.notna(p) else np.array([], dtype=int),
            np.array([int(s)]) if pd.notna(s) else np.array([], dtype=int),
        )
    if len(types) != len(arrivals):
        return np.array([], dtype=int), np.array([], dtype=int)
    p_arr = [int(arrivals[i]) for i, t in enumerate(types) if t == "P"]
    s_arr = [int(arrivals[i]) for i, t in enumerate(types) if t == "S"]
    return np.array(p_arr, dtype=int), np.array(s_arr, dtype=int)


# ---------------------------------------------------------------------------
# Waveform processing
# ---------------------------------------------------------------------------
def pad_and_slice(wf_3C, p_sample, s_sample, all_p, all_s):
    """Pad with noise and random-slice to DATA_NPTS, return (wf, valid_p, valid_s)."""
    if wf_3C.shape[0] == 3:
        wf_3C = wf_3C.T
    npts = wf_3C.shape[0]
    ps_res = s_sample - p_sample
    max_p_pos = max(min(DATA_NPTS - ps_res - 1000, DATA_NPTS - 2000), MIN_P_POSITION + 100)
    front_pad = max_p_pos + 1000
    back_pad = DATA_NPTS + 1000
    padded = np.zeros((front_pad + npts + back_pad, 3), dtype=wf_3C.dtype)

    for ch in range(3):
        data = wf_3C[:, ch]
        pre_p = data[: min(p_sample, len(data))]
        if len(pre_p) >= 500:
            ref = find_reference_signal(pre_p, 500, len(pre_p), 100)
        elif len(pre_p) >= 200:
            ref = find_reference_signal(pre_p, 200, len(pre_p), 50)
        else:
            ref = pre_p if len(pre_p) > 0 else data[:500]
        tail_len = min(500, len(data) - s_sample - 100)
        back_ref = data[-tail_len:] if tail_len >= 200 else ref
        padded[:, ch] = np.concatenate(
            [
                generate_matching_noise(ref, front_pad),
                data,
                generate_matching_noise(back_ref, back_pad),
            ]
        )

    target_p = np.random.randint(MIN_P_POSITION, max_p_pos + 1)
    slice_start = front_pad + p_sample - target_p
    sliced = padded[slice_start : slice_start + DATA_NPTS]

    offset = target_p - p_sample
    valid_p = all_p + offset
    valid_s = all_s + offset
    valid_p = valid_p[(valid_p >= 0) & (valid_p < DATA_NPTS)]
    valid_s = valid_s[(valid_s >= 0) & (valid_s < DATA_NPTS)]
    return sliced, valid_p, valid_s


def get_ps_diff_bin(ps_diff_samples):
    """Bin P-S diff into sampling category."""
    ps_sec = ps_diff_samples * DT
    if ps_sec < 5:
        return "singleEQ_00-05s"
    elif ps_sec < 10:
        return "singleEQ_05-10s"
    elif ps_sec < 15:
        return "singleEQ_10-15s"
    elif ps_sec < 20:
        return "singleEQ_15-20s"
    else:
        return "singleEQ_20s_plus"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def convert(
    ceed_dir: Path,
    out_dir: Path,
    region: str,
    max_per_year=None,
    shard_size=50000,
    legacy_shard_index=False,
) -> dict:
    """Write the shards of one region; returns {split: samples written}."""
    rcfg = REGION_CONFIG[region]
    prefix = rcfg["prefix"]
    years = rcfg["years"]
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info(f"CEED {prefix.upper()} -> RED-PAN H5 (direct, no TFRecords)")
    log.info(f"  Years: {len(years)} year-keys")
    log.info(f"  Output: {out_dir}")
    log.info("=" * 70)

    # Accumulate per split
    split_data = defaultdict(lambda: {"waveforms": [], "meta": [], "shard_idx": 0, "sample_idx": 0})
    next_shard = [0]  # shard number shared by all splits (unless legacy_shard_index)

    def flush_shard(split):
        sd = split_data[split]
        if not sd["waveforms"]:
            return
        n = len(sd["waveforms"])
        shard = sd["shard_idx"] if legacy_shard_index else next_shard[0]
        h5_name = f"CEED_{prefix.upper()}_singleEQ_s{shard:03d}.h5"
        csv_name = f"CEED_{prefix.upper()}_singleEQ_s{shard:03d}_metadata.csv"
        h5_path = out_dir / h5_name
        csv_path = out_dir / csv_name

        wf_arr = np.stack(sd["waveforms"], axis=0)  # (N, 3, 9000)
        with h5py.File(h5_path, "w") as hf:
            grp = hf.create_group("singleEQ")
            split_grp = grp.create_group(split)
            split_grp.create_dataset(
                "waveforms",
                data=wf_arr,
                dtype=np.float32,
                chunks=(min(64, n), 3, DATA_NPTS),
                compression="gzip",
                compression_opts=4,
            )

        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "hdf5_index", "sample_id", "category", "sampling_category", "split",
                    "source_file", "p_arrival_sample", "s_arrival_sample", "ps_diff_samples",
                    "lightweight_storage",
                ],
            )  # fmt: skip
            writer.writeheader()
            for i, row in enumerate(sd["meta"]):
                row["hdf5_index"] = i  # per-shard 0-based index
                writer.writerow(row)

        log.info(f"  Shard {h5_name}: {n} samples ({split})")
        sd["waveforms"].clear()
        sd["meta"].clear()
        sd["shard_idx"] += 1
        next_shard[0] += 1

    total = 0
    errors = 0

    for year in years:
        csv_path = ceed_dir / f"metadata{prefix}{year}.csv"
        h5_path = ceed_dir / f"waveforms{prefix}{year}.hdf5"
        if not csv_path.exists() or not h5_path.exists():
            log.warning(f"Skipping year {year}: files not found")
            continue

        df = pd.read_csv(csv_path, low_memory=False)
        df = df.dropna(subset=["trace_p_arrival_sample", "trace_s_arrival_sample"])
        df["trace_p_arrival_sample"] = df["trace_p_arrival_sample"].astype(int)
        df["trace_s_arrival_sample"] = df["trace_s_arrival_sample"].astype(int)
        df = df[df["trace_s_arrival_sample"] > df["trace_p_arrival_sample"] + MIN_PS_RESIDUAL]
        df = df[df["trace_s_arrival_sample"] < ORIGINAL_NPTS]

        if max_per_year and len(df) > max_per_year:
            df = df.sample(n=max_per_year, random_state=42)

        # Determine split
        if "split" in df.columns:
            df["_split"] = df["split"].map(CEED_SPLIT_MAP).fillna("train")
        else:
            df["_split"] = "train"

        log.info(f"Year {year}: {len(df)} valid EQ traces")

        hf = h5py.File(h5_path, "r")
        for _, row in df.iterrows():
            trace_name = row["trace_name"]
            parts = trace_name.split("/")
            if len(parts) != 2:
                errors += 1
                continue
            try:
                wf = np.array(hf[parts[0]][parts[1]], dtype=np.float32)  # (3, 12000)
            except KeyError:
                errors += 1
                continue
            if wf.shape != (3, ORIGINAL_NPTS):
                errors += 1
                continue

            p_sample = int(row["trace_p_arrival_sample"])
            s_sample = int(row["trace_s_arrival_sample"])
            all_p, all_s = parse_all_ps_samples(row)

            try:
                sliced, valid_p, valid_s = pad_and_slice(wf, p_sample, s_sample, all_p, all_s)
            except Exception:
                errors += 1
                continue

            if sliced.shape != (DATA_NPTS, 3):
                errors += 1
                continue

            split = row["_split"]
            sd = split_data[split]

            # Store as (3, 9000) channel-first
            sd["waveforms"].append(sliced.T.astype(np.float32))

            ps_diff = int(valid_s[0] - valid_p[0]) if len(valid_p) > 0 and len(valid_s) > 0 else 0
            sd["meta"].append(
                {
                    "hdf5_index": sd["sample_idx"],
                    "sample_id": f"singleEQ_{trace_name.replace('/', '_')}",
                    "category": "singleEQ",
                    "sampling_category": get_ps_diff_bin(ps_diff),
                    "split": split,
                    "source_file": f"{prefix}{year}",
                    "p_arrival_sample": str(valid_p.tolist()),
                    "s_arrival_sample": str(valid_s.tolist()),
                    "ps_diff_samples": ps_diff,
                    "lightweight_storage": True,
                }
            )
            sd["sample_idx"] += 1
            total += 1

            # Flush shard if needed
            if len(sd["waveforms"]) >= shard_size:
                flush_shard(split)

            if total % 50000 == 0:
                log.info(f"  Progress: {total:,} samples, {errors} errors")

        hf.close()

    # Flush remaining
    for split in split_data:
        flush_shard(split)

    log.info("=" * 70)
    log.info(f"DONE. Total: {total:,} samples, {errors} errors")
    for split, sd in split_data.items():
        log.info(f"  {split}: {sd['sample_idx']:,} samples, {sd['shard_idx']} shards")
    log.info(f"Output: {out_dir}")
    log.info("=" * 70)
    return {split: sd["sample_idx"] for split, sd in split_data.items()}


def main():
    p = argparse.ArgumentParser(
        prog="python -m builder.ceed_convert",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=None,
        help="YAML file with data locations (default: $RPM_BENCH_CONFIG; RPM_BENCH_<KEY> overrides)",
    )
    p.add_argument("--region", choices=["nc", "sc"], required=True)
    p.add_argument("--ceed-dir", type=Path, default=None, help="default: config key ceed_cache")
    p.add_argument(
        "--out-dir", type=Path, default=None, help="default: config key ceed_<region>_h5_dir"
    )
    p.add_argument("--max-per-year", type=int, default=None, help="cap per year (smoke runs)")
    p.add_argument("--shard-size", type=int, default=50000, help="samples per H5 shard")
    p.add_argument(
        "--legacy-shard-index",
        action="store_true",
        help="number shards per split as the original run did (later splits overwrite earlier "
        "shards with the same number)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="seed numpy's global RNG (default: unseeded)"
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s : %(asctime)s : %(message)s")
    cfg = sources.load(args.config)
    ceed_dir = args.ceed_dir or cfg["ceed_cache"]
    out_dir = args.out_dir or cfg[f"ceed_{args.region}_h5_dir"]
    if not ceed_dir.is_dir():
        raise SystemExit(f"CEED cache not found: {ceed_dir} (config key ceed_cache)")
    if args.seed is not None:
        np.random.seed(args.seed)
    counts = convert(
        ceed_dir, out_dir, args.region, args.max_per_year, args.shard_size, args.legacy_shard_index
    )
    if not any(counts.values()):
        raise SystemExit(f"no CEED traces converted from {ceed_dir}")


if __name__ == "__main__":
    main()
