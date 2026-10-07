#!/usr/bin/env python
"""Build the GeoNet 2013-2014 holdout used by Table III, Table IV and Fig. 3 (not used in training).

Source: the GeoNet benchmark dataset (``geonet_source_root``):
  event_dataset/metadata/metadata_events_<year>.csv, event_dataset/waveform_data/{units,counts}/waveforms_*_<year>.h5
  noise_dataset/metadata/metadata_noise.csv,         noise_dataset/waveform_data/{units,counts}/waveforms_*_noise.h5
Output (``geonet_holdout_root``):
  --kind events: waveforms/<trace>.npy (27000, 3) with P at sample 9000 (spectrum-matched padding
                 front and back), metadata.csv (p_sample, s_sample, ...); 2013 + 2014 earthquakes with
                 P and S inside the 120 s record (70,583 records in the manuscript).
  --kind noise:  noise_waveforms/<trace>.npy (12000, 3) unpadded, metadata_noise.csv; the 15% of noise
                 records left out by a seeded split (pandas sample frac 0.85, random_state 42), minus
                 records with a 10 s window std ratio > 10 (22,479 records in the manuscript).
Copied from the scripts that built the manuscript's set (QC figures removed). The event padding is not
seeded (``--seed`` seeds it for a repeatable rebuild), so a rebuild is equivalent but not bit-identical
to the manuscript's set.
"""

from __future__ import annotations

import argparse
import logging
import os

import h5py
import numpy as np
import pandas as pd
from rpm_bench import config

from redpan_motion.utils.waveform import find_reference_signal, generate_matching_noise

ORIGINAL_NPTS, OUTPUT_NPTS, P_TARGET, DT = 12000, 27000, 9000, 0.01


def categorize_ps_residual(ps_sec):
    """Categorize PS residual into groups."""
    if ps_sec < 5:
        return "<5s"
    elif ps_sec < 10:
        return "5-10s"
    elif ps_sec < 15:
        return "10-15s"
    elif ps_sec < 20:
        return "15-20s"
    else:
        return ">20s"


def pad_waveform_for_benchmark(wf_3C, p_orig, s_orig, output_npts=27000, p_target=9000):
    """
    Pad waveform so P arrival is at p_target sample.
    Front: spectrum-matched noise from pre-P data.
    Back: spectrum-matched noise from tail data.
    Total output length: output_npts.

    Parameters
    ----------
    wf_3C : np.ndarray
        Waveform array, shape (npts, 3)
    p_orig : int
        P arrival sample in original waveform
    s_orig : int
        S arrival sample in original waveform
    output_npts : int
        Total output length in samples (default: 27000 = 270s)
    p_target : int
        Target P arrival position in output (default: 9000 = 90s)

    Returns
    -------
    output : np.ndarray
        Padded waveform, shape (output_npts, 3), float32
    new_p : int
        P arrival sample in output
    new_s : int
        S arrival sample in output
    """
    npts, n_ch = wf_3C.shape
    ps_residual = s_orig - p_orig
    output = np.zeros((output_npts, n_ch), dtype=np.float32)

    if p_orig <= p_target:
        # Need front padding
        front_pad = p_target - p_orig
        data_start_out = front_pad
        data_slice = wf_3C
    else:
        # Trim front (P is beyond 90s in original)
        front_pad = 0
        trim = p_orig - p_target
        data_start_out = 0
        data_slice = wf_3C[trim:]

    data_len = len(data_slice)
    data_end_out = data_start_out + data_len

    # Clamp if data exceeds output length
    if data_end_out > output_npts:
        data_len = output_npts - data_start_out
        data_end_out = output_npts
        data_slice = data_slice[:data_len]

    # Place original data
    output[data_start_out:data_end_out, :] = data_slice.astype(np.float32)

    for ch in range(n_ch):
        # Front padding: spectrum-matched noise from pre-P waveform
        if front_pad > 0:
            pre_p_data = wf_3C[:p_orig, ch]
            if len(pre_p_data) >= 500:
                ref = find_reference_signal(
                    pre_p_data, window_size=500, max_search=len(pre_p_data), min_unique=100
                )
            elif len(pre_p_data) >= 200:
                ref = find_reference_signal(
                    pre_p_data, window_size=200, max_search=len(pre_p_data), min_unique=50
                )
            else:
                ref = pre_p_data if len(pre_p_data) > 0 else wf_3C[:500, ch]
            output[:front_pad, ch] = generate_matching_noise(ref, front_pad)

        # Back padding: spectrum-matched noise from tail of waveform
        back_pad = output_npts - data_end_out
        if back_pad > 0:
            tail_data = wf_3C[-500:, ch]
            if np.std(tail_data) > 0:
                ref = tail_data
            else:
                # Fallback to pre-P if tail is flat
                ref = wf_3C[: min(500, p_orig), ch]
                if len(ref) == 0 or np.std(ref) == 0:
                    ref = wf_3C[:500, ch]
            output[data_end_out:, ch] = generate_matching_noise(ref, back_pad)

    new_p = p_target
    new_s = p_target + ps_residual

    return output, new_p, new_s


def open_h5(units, counts):
    """Open the units waveform file, or the counts file if that cannot be opened; logs which."""
    try:
        h = h5py.File(units, "r")
    except OSError:
        logging.info("cannot open %s; reading %s", units, counts)
        return h5py.File(counts, "r")
    logging.info("reading %s", units)
    return h


def build_events(basedir, outpath):
    """Write the padded 2013-2014 earthquake records and metadata.csv."""
    wf_outdir = os.path.join(outpath, "waveforms")
    os.makedirs(wf_outdir, exist_ok=True)
    test_years = [2013, 2014]
    dfs = []
    for year in test_years:
        d = pd.read_csv(
            os.path.join(basedir, "event_dataset", "metadata", f"metadata_events_{year}.csv")
        )
        d["year"] = year
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df_eq = df[df["trace_category"] == "earthquake"].copy()
    df_eq = df_eq.dropna(subset=["trace_p_arrival_sample", "trace_s_arrival_sample"])
    df_eq["trace_p_arrival_sample"] = df_eq["trace_p_arrival_sample"].astype(int)
    df_eq["trace_s_arrival_sample"] = df_eq["trace_s_arrival_sample"].astype(int)
    df_eq = df_eq[df_eq["trace_s_arrival_sample"] > df_eq["trace_p_arrival_sample"]]
    df_eq = df_eq[df_eq["trace_s_arrival_sample"] < ORIGINAL_NPTS]
    df_eq["ps_residual_sec"] = (
        df_eq["trace_s_arrival_sample"] - df_eq["trace_p_arrival_sample"]
    ) * DT
    df_eq["ps_group"] = df_eq["ps_residual_sec"].apply(categorize_ps_residual)
    wf_units = os.path.join(basedir, "event_dataset", "waveform_data", "units")
    wf_counts = os.path.join(basedir, "event_dataset", "waveform_data", "counts")
    records, skipped = [], 0
    for year in test_years:
        df_year = df_eq[df_eq["year"] == year]
        if len(df_year) == 0:
            continue
        with open_h5(
            os.path.join(wf_units, f"waveforms_units_{year}.h5"),
            os.path.join(wf_counts, f"waveforms_counts_{year}.h5"),
        ) as hf:
            for _, row in df_year.iterrows():
                trace_name = row["trace_name"]
                p_orig, s_orig = (
                    int(row["trace_p_arrival_sample"]),
                    int(row["trace_s_arrival_sample"]),
                )
                try:
                    wf_data = hf["data"].get(trace_name)
                    if wf_data is None:
                        skipped += 1
                        continue
                    wf_3C = np.array(wf_data).T
                    if wf_3C.shape != (ORIGINAL_NPTS, 3):
                        skipped += 1
                        continue
                    output_wf, new_p, new_s = pad_waveform_for_benchmark(
                        wf_3C, p_orig, s_orig, output_npts=OUTPUT_NPTS, p_target=P_TARGET
                    )
                    if new_s >= OUTPUT_NPTS:
                        skipped += 1
                        continue
                    np.save(os.path.join(wf_outdir, f"{trace_name}.npy"), output_wf)
                    records.append(
                        {
                            "trace_name": trace_name,
                            "year": year,
                            "p_sample": new_p,
                            "s_sample": new_s,
                            "p_sample_original": p_orig,
                            "s_sample_original": s_orig,
                            "ps_residual_sec": round((s_orig - p_orig) * DT, 4),
                            "ps_group": row["ps_group"],
                            "source_magnitude": row.get("source_magnitude", np.nan),
                            "path_hyp_distance_km": row.get("path_hyp_distance_km", np.nan),
                            "source_depth_km": row.get("source_depth_km", np.nan),
                            "station_code": row.get("station_code", ""),
                            "station_channels": row.get("station_channels", ""),
                        }
                    )
                except Exception as e:  # noqa: BLE001 - as when the set was built: log and skip the trace
                    logging.warning("error processing %s: %s", trace_name, e)
                    skipped += 1
    pd.DataFrame(records).to_csv(os.path.join(outpath, "metadata.csv"), index=False)
    logging.info("events: %d written, %d skipped -> %s", len(records), skipped, outpath)


def build_noise(basedir, outpath, snr_threshold=10):
    """Write the held-out noise records and metadata_noise.csv."""
    wf_outdir = os.path.join(outpath, "noise_waveforms")
    os.makedirs(wf_outdir, exist_ok=True)
    df = pd.read_csv(os.path.join(basedir, "noise_dataset", "metadata", "metadata_noise.csv"))
    df_noise = df[df["trace_category"] == "noise"].copy()
    df_train = df_noise.sample(frac=0.85, random_state=42)
    df_use = df_noise[~df_noise.index.isin(df_train.index)]
    records, skipped, snr_skipped = [], 0, 0
    wd = os.path.join(basedir, "noise_dataset", "waveform_data")
    with open_h5(
        os.path.join(wd, "units", "waveforms_units_noise.h5"),
        os.path.join(wd, "counts", "waveforms_counts_noise.h5"),
    ) as hf:
        for _, row in df_use.iterrows():
            trace_name = row["trace_name"]
            try:
                wf_data = hf["data"].get(trace_name)
                if wf_data is None:
                    skipped += 1
                    continue
                wf_3C = np.array(wf_data).T
                if wf_3C.shape != (ORIGINAL_NPTS, 3):
                    skipped += 1
                    continue
                z_data = wf_3C[:, 2]
                win = 1000
                n_wins = len(z_data) // win
                if n_wins >= 2:
                    stds = np.array(
                        [np.std(z_data[i * win : (i + 1) * win]) for i in range(n_wins)]
                    )
                    if np.min(stds) > 0 and np.max(stds) / np.min(stds) > snr_threshold:
                        snr_skipped += 1
                        continue
                np.save(os.path.join(wf_outdir, f"{trace_name}.npy"), wf_3C.astype(np.float32))
                records.append(
                    {
                        "trace_name": trace_name,
                        "station_code": row.get("station_code", ""),
                        "station_channels": row.get("station_channels", ""),
                        "output_npts": ORIGINAL_NPTS,
                        "category": "noise",
                    }
                )
            except Exception as e:  # noqa: BLE001 - as when the set was built: log and skip the trace
                logging.warning("error processing %s: %s", trace_name, e)
                skipped += 1
    pd.DataFrame(records).to_csv(os.path.join(outpath, "metadata_noise.csv"), index=False)
    logging.info(
        "noise: %d written, %d skipped, %d removed by the std-ratio check -> %s",
        len(records),
        skipped,
        snr_skipped,
        outpath,
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument("--kind", required=True, choices=["events", "noise"])
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed numpy's global RNG, which draws the event padding (default: unseeded, as for "
        "the manuscript's set; the noise split is always seeded)",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(asctime)s : %(message)s")
    if args.seed is not None:
        np.random.seed(args.seed)
    cfg = config.load(args.config)
    src, out = str(cfg["geonet_source_root"]), str(cfg["geonet_holdout_root"])
    (build_events if args.kind == "events" else build_noise)(src, out)


if __name__ == "__main__":
    main()
