#!/usr/bin/env python
"""Streaming replay (Discussion): 437 Taiwan earthquakes and 4,370 noise records of 120 s.

Predictions are updated every 0.05 s. Earthquakes: windows ending from 1 s before to 5 s after the
labeled P (SAC header ``t1``). Noise: windows ending from one window length after the record start
to its end. For every window, P peaks >= 0.1 inside ``trigger_onset(P, 0.05, 0.05)`` regions are kept
(earthquakes: within 0.5 s of the labeled P); the delay is window end minus pick time.

Writes one tab-separated CSV per event to ``<results_root>/streaming/<model>/<mode>/<EVID>.csv`` with
columns ``inference_endtime, delay, P_probability, Mask_probability, station_id`` (noise EVIDs are
``<network>_<dir>``). ``score_streaming.py`` scores them. Each file is written to ``<EVID>.csv.tmp``
and renamed when complete, so an interrupted run resumes by skipping the finished events. The
60 s models' ``--noise-mask mask`` variant goes to ``.../noise_mask/`` so it does not mix with the
published (legacy) noise replay.

Inputs (``eew_root``): ``EQ_Mw4_to_6.9/<TSMIP|Palert>_*/*.sac`` and
``noise/sac_120s_{CWB_StrongMotion,Palert}/<5-char id>/*.sac``.

The 90 s models (edge, rpm) and the 60 s RED-PAN (redpan, redpan_240107) use the two original replay
codes (see rpm_bench/streaming_utils.py); both read raw, unfiltered data.

Example:
    python run_streaming.py --model edge --mode eq
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import defaultdict
from copy import deepcopy
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.trigger import trigger_onset
from rpm_bench import config, models
from rpm_bench.streaming_utils import (
    DELAY_THRE,
    DT,
    LOOP_POST_P_SEC,
    LOOP_PRE_P_SEC,
    P_DETECT_THRE,
    P_TRUE_PICK_THRE,
    PRED_INTERVAL_SEC,
    TRIGGER_MAX_LEN,
    TRIGGER_ON_OFF,
    compute_pre_p_medians,
    list_eq_events,
    list_noise_subdirs,
    pad_and_zscore,
    predict_60s,
    predict_batch_90s,
    process_station_windows,
    sac_len_complement,
    stream_standardize,
    stream_to_zraw,
)
from scipy.signal import find_peaks


# ──────────────────────────────────────────────────────────────────────────
# 90 s models (eew_dynamic_pred_v49.py)
# ──────────────────────────────────────────────────────────────────────────
def collect_station_windows_eq(datadir, sta_ids, pred_npts=9000):
    """Windows ending from 1 s before to 5 s after P, every 0.05 s, for each station of an event.
    Returns (package or None, number of stations skipped for too little data around P)."""
    pred_sec = pred_npts // 100
    ev_wid = []
    ev_stt = []
    ev_ent = []
    ev_data = []
    ev_zraw = []
    ev_gt_p = []
    sta_n_slice = []
    _ct = 0
    n_fail = 0
    for sta in sta_ids:
        read_idx = os.path.join(datadir, f"{sta}*.sac")
        sl = read(read_idx)
        if abs(sl[0].stats.sampling_rate - 100) > 0.01:
            sl = sl.resample(100)
        sl.sort()
        info = sl[0].stats
        gt_p = info.starttime + info.sac.t1 - info.sac.b
        p_npts = int((gt_p - info.starttime) / DT)
        if p_npts < 100 or (info.endtime - gt_p) < 5:
            n_fail += 1
            continue
        pad_vals = compute_pre_p_medians(sl, gt_p)
        pad_z = float(pad_vals[2])
        n = int(np.round(((gt_p + LOOP_POST_P_SEC) - (gt_p - LOOP_PRE_P_SEC)) / PRED_INTERVAL_SEC))
        wf_ent = np.array([gt_p - LOOP_PRE_P_SEC + PRED_INTERVAL_SEC * i for i in range(n)])
        wf_stt = wf_ent - pred_sec
        data = np.stack(
            [pad_and_zscore(sl, wf_stt[w], wf_ent[w], pad_vals) for w in range(n)], axis=0
        )
        z_raw = np.stack(
            [stream_to_zraw(sl, wf_stt[w], wf_ent[w], pad_z) for w in range(n)], axis=0
        )
        wid = np.repeat(sl[0].id[:-1], n)
        ev_wid.append(wid)
        ev_stt.append(wf_stt)
        ev_ent.append(wf_ent)
        ev_data.append(data)
        ev_zraw.append(z_raw)
        ev_gt_p.append(gt_p)
        sta_n_slice.append((_ct, _ct + n))
        _ct += n
    if not ev_wid:
        return None, n_fail
    return dict(
        wid=np.concatenate(ev_wid),
        stt=np.concatenate(ev_stt),
        ent=np.concatenate(ev_ent),
        data=np.concatenate(ev_data, axis=0),
        zraw=np.concatenate(ev_zraw, axis=0),
        gt_p=ev_gt_p,
        slices=sta_n_slice,
    ), n_fail


def collect_station_windows_noise(datadir, sta_ids, pred_npts=9000):
    """Windows ending every 0.05 s from one window length after the start to the end of each record."""
    pred_sec = pred_npts // 100
    ev_wid = []
    ev_stt = []
    ev_ent = []
    ev_data = []
    ev_zraw = []
    sta_n_slice = []
    _ct = 0
    for sta in sta_ids:
        sl = read(os.path.join(datadir, f"*.{sta}.*.sac"))
        sl.sort()
        info = sl[0].stats
        start_utc = info.starttime + pred_sec
        end_utc = info.endtime
        n = int(np.round((end_utc - start_utc) / PRED_INTERVAL_SEC))
        if n <= 0:
            continue
        pad_vals = compute_pre_p_medians(sl, gt_p_utc=None)
        pad_z = float(pad_vals[2])
        wf_ent = np.array([start_utc + PRED_INTERVAL_SEC * i for i in range(n)])
        wf_stt = wf_ent - pred_sec
        data = np.stack(
            [pad_and_zscore(sl, wf_stt[w], wf_ent[w], pad_vals) for w in range(n)], axis=0
        )
        z_raw = np.stack(
            [stream_to_zraw(sl, wf_stt[w], wf_ent[w], pad_z) for w in range(n)], axis=0
        )
        wid = np.repeat(sl[0].id[:-1], n)
        ev_wid.append(wid)
        ev_stt.append(wf_stt)
        ev_ent.append(wf_ent)
        ev_data.append(data)
        ev_zraw.append(z_raw)
        sta_n_slice.append((_ct, _ct + n))
        _ct += n
    if not ev_wid:
        return None
    return dict(
        wid=np.concatenate(ev_wid),
        stt=np.concatenate(ev_stt),
        ent=np.concatenate(ev_ent),
        data=np.concatenate(ev_data, axis=0),
        zraw=np.concatenate(ev_zraw, axis=0),
        slices=sta_n_slice,
    )


def run_event_90s(model, mode, evdir, device, batch_size):
    """Replay one event (or noise directory) with a 90 s model; per-window rows, or None."""
    if mode == "eq":
        sta_ids = np.unique([os.path.basename(p)[:-5] for p in glob(os.path.join(evdir, "*.sac"))])
        pkg, n_fail = collect_station_windows_eq(evdir, sta_ids)
        if n_fail:
            logging.info(
                "%s: %d of %d stations skipped (fewer than 100 samples before P or less "
                "than 5 s after it)",
                os.path.basename(evdir),
                n_fail,
                len(sta_ids),
            )
    else:
        sta_ids = np.unique(
            [os.path.basename(p).split(".")[1] for p in glob(os.path.join(evdir, "*.sac"))]
        )
        pkg = collect_station_windows_noise(evdir, sta_ids)
    if pkg is None:
        return None
    P_all, D_all = predict_batch_90s(model, pkg["data"], pkg["zraw"], device, batch_size=batch_size)
    rows = []
    for i, (lo, hi) in enumerate(pkg["slices"]):
        gt_p = pkg["gt_p"][i] if mode == "eq" else None
        station_dict = process_station_windows(
            P_all[lo:hi], D_all[lo:hi], pkg["stt"][lo:hi], pkg["ent"][lo:hi], gt_p
        )
        df = (
            pd.DataFrame.from_dict(station_dict, orient="index")
            .rename_axis("inference_endtime")
            .reset_index()
        )
        df["station_id"] = pkg["wid"][lo]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────
# 60 s RED-PAN (P01_tf60_*.py = eq, P02_tf60_*.py = noise)
# ──────────────────────────────────────────────────────────────────────────
def run_event_60s(model, mode, evdir, device, pred_npts=6000, noise_mask="legacy"):
    """Replay one event (or noise directory) with a 60 s RED-PAN; per-window rows, or None.
    The thresholds are the streaming_utils constants (the P01 / P02 scripts declared the same
    values locally)."""
    pred_sec = int(pred_npts * DT)
    detect_dict = defaultdict(dict)
    if mode == "eq":
        evid = os.path.basename(evdir)
        sta_id = np.unique([os.path.basename(i)[:-5] for i in glob(os.path.join(evdir, "*.sac"))])
    else:
        network = os.path.split(evdir)[0].split("_")[-1]
        evid = f"{network}_{os.path.basename(evdir)}"
        sta_id = np.unique(
            [os.path.basename(i).split(".")[1] for i in glob(os.path.join(evdir, "*.sac"))]
        )
    detect_dict[evid] = defaultdict(dict)

    ev_wid, ev_wf_stt, ev_wf_ent, ev_wf_data, ev_p_utc, sta_n_slice_idx = [], [], [], [], [], []
    _ct = 0
    n_windows = n_short = 0
    for ct, p in enumerate(sta_id):
        if mode == "eq":
            read_idx = os.path.join(evdir, f"{p}*.sac")
            # read once (the original read the same files up to three times; same stream)
            st = read(read_idx)
            info = st[0].stats
            if info.sampling_rate != 100:
                wf = sac_len_complement(st.resample(100))
            else:
                wf = sac_len_complement(st)
            wf.sort()
            info = wf[0].stats
            p_utc = info.starttime + info.sac.t1 - info.sac.b
            p_npts = int((p_utc - info.starttime) / DT)
            if p_npts < 100 or (info.endtime - p_utc < 5):
                continue
            # collect waveform 1 second before and 5 seconds after labeled P
            n_slices = np.round(
                ((p_utc + LOOP_POST_P_SEC) - (p_utc - LOOP_PRE_P_SEC)) / PRED_INTERVAL_SEC
            ).astype(int)
            wf_ent = np.sort(
                [p_utc - LOOP_PRE_P_SEC + PRED_INTERVAL_SEC * N for N in range(n_slices)]
            )
        else:
            wf = read(os.path.join(evdir, f"*.{p}.*.sac"))
            wf.sort()
            info = wf[0].stats
            start_utc = info.starttime + pred_sec
            end_utc = info.endtime
            n_slices = np.round((end_utc - start_utc) / PRED_INTERVAL_SEC).astype(int)
            wf_ent = np.sort([start_utc + PRED_INTERVAL_SEC * N for N in range(n_slices)])
            p_utc = None
        wf_stt = wf_ent - pred_sec
        wf_collect = []
        for W in range(len(wf_stt)):
            win = deepcopy(wf).slice(wf_stt[W], wf_ent[W])
            # windows that start before the record are short; stream_standardize end-pads them
            # (README, Known caveats). Counted only; the processing is unchanged.
            n_short += any(len(tr.data) < pred_npts for tr in win)
            wf_collect.append(stream_standardize(win, pred_npts))
        n_windows += len(wf_stt)
        get_data = lambda x: np.stack([x[0].data, x[1].data, x[2].data], -1)
        wf_data = np.stack([get_data(x) for x in wf_collect], 0)
        wid = np.repeat(wf[0].id[:-1], n_slices)
        ev_wid.append(wid)
        ev_wf_stt.append(wf_stt)
        ev_wf_ent.append(wf_ent)
        ev_wf_data.append(wf_data)
        ev_p_utc.append(p_utc)
        sta_n_slice_idx.append([_ct, _ct + n_slices])
        _ct += n_slices
    if not ev_wid:
        return None
    if n_short:
        logging.info(
            "%s: %d of %d windows shorter than %d samples (end-padded by stream_standardize)",
            evid,
            n_short,
            n_windows,
            pred_npts,
        )

    ev_wf_data = np.vstack(ev_wf_data)
    Pick, Mask = predict_60s(model, ev_wf_data, device)
    predP = Pick[..., 0]
    predM = Mask[..., 0]
    delay_df_sta = []
    for i in range(len(ev_wid)):
        e_id = ev_wid[i]
        e_stt = ev_wf_stt[i]
        e_ent = ev_wf_ent[i]
        e_p_utc = ev_p_utc[i]
        P_func = predP[sta_n_slice_idx[i][0] : sta_n_slice_idx[i][1]]
        M_func = predM[sta_n_slice_idx[i][0] : sta_n_slice_idx[i][1]]
        if e_id[0] not in detect_dict[evid]:
            detect_dict[evid][e_id[0]] = defaultdict(dict)
        for j in range(len(P_func)):
            inf_id = str(e_ent[j])[:22]
            if inf_id not in detect_dict[evid][e_id[0]]:
                detect_dict[evid][e_id[0]][inf_id] = {
                    "delay": -1,
                    "P_probability": -1,
                    "Mask_probability": -1,
                }
            rt_p = P_func[j]
            rt_M = M_func[j]
            search_range = trigger_onset(
                rt_p, TRIGGER_ON_OFF[0], TRIGGER_ON_OFF[1], max_len=TRIGGER_MAX_LEN
            )
            p_peaks, p_peaks_info = find_peaks(rt_p, height=P_DETECT_THRE, distance=int(1 / DT))
            p_peaks_confidence = p_peaks_info["peak_heights"]
            bool_p_peaks = []
            # range(s[0], s[1]) leaves out trigger_onset's inclusive end sample (README, Known caveats)
            for _p in p_peaks:
                check_trgP = len(
                    np.where(np.array([_p in range(s[0], s[1]) for s in search_range]))[0]
                )
                bool_p_peaks.append(check_trgP > 0)
            if len(bool_p_peaks) > 0:
                p_peaks = p_peaks[np.array(bool_p_peaks)]
                p_rt_pb = p_peaks_confidence[np.array(bool_p_peaks)]
            else:
                p_peaks = []
                p_rt_pb = []
            if len(p_peaks) < 1:
                continue
            for t in range(len(p_peaks)):
                trg_peak = p_peaks[t]
                trg_pb = p_rt_pb[t]
                mask_pb = np.mean(rt_M[trg_peak:])
                trg_p_utc = e_stt[j] + trg_peak * DT
                delay = e_ent[j] - trg_p_utc
                if mode == "eq" and np.abs(trg_p_utc - e_p_utc) > P_TRUE_PICK_THRE:
                    continue
                if delay > DELAY_THRE:
                    continue
                detect_dict[evid][e_id[0]][inf_id]["delay"] = delay
                detect_dict[evid][e_id[0]][inf_id]["P_probability"] = trg_pb
                # The manuscript's 60 s noise replay (P02) stored the P probability in this column
                # (mask_pb was computed but unused); "legacy" keeps that so the published values
                # reproduce, "mask" stores the mask as the earthquake replay and the 90 s models do.
                use_mask = mode == "eq" or noise_mask == "mask"
                detect_dict[evid][e_id[0]][inf_id]["Mask_probability"] = (
                    mask_pb if use_mask else trg_pb
                )
        delay_df = (
            pd.DataFrame.from_dict(detect_dict[evid][e_id[0]], "index")
            .rename_axis("inference_endtime")
            .reset_index()
        )
        delay_df["station_id"] = e_id[0]
        delay_df_sta.append(delay_df)
    return pd.concat(delay_df_sta)


def default_out_subdir(mode: str, noise_mask: str, is60: bool) -> str:
    """``eq`` / ``noise``; ``noise_mask`` for the 60 s models' ``--noise-mask mask`` variant."""
    return "noise_mask" if (mode == "noise" and noise_mask == "mask" and is60) else mode


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument("--model", required=True, choices=models.MODEL_KEYS)
    p.add_argument("--mode", required=True, choices=["eq", "noise"])
    p.add_argument(
        "--max-events", type=int, default=0, help="first N events / noise dirs (0 = all)"
    )
    p.add_argument(
        "--batch-size", type=int, default=64, help="forward batch size for the 90 s models"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default <results_root>/streaming/<model>/<mode> (noise_mask for --noise-mask mask "
        "with a 60 s model)",
    )
    p.add_argument(
        "--noise-mask",
        default="legacy",
        choices=["legacy", "mask"],
        help="60 s models, noise mode: 'legacy' stores the P probability in Mask_probability, as the "
        "published replay did; 'mask' stores the mean mask after the pick (as for earthquakes)",
    )
    p.add_argument(
        "--device",
        default=models.default_device(),
        help="torch device (default: cuda:0 if available, else cpu)",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = config.load(args.config)
    ev_root = cfg["eew_root"] / ("EQ_Mw4_to_6.9" if args.mode == "eq" else "noise")
    events = list_eq_events(str(ev_root)) if args.mode == "eq" else list_noise_subdirs(str(ev_root))
    if len(events) == 0:
        raise SystemExit(
            f"no {args.mode} events found under {ev_root}; check config key eew_root / "
            f"RPM_BENCH_EEW_ROOT"
        )
    if args.max_events > 0:
        events = events[: args.max_events]
    is60 = args.model in models.SIXTY_S
    out_dir = args.out_dir or config.results_dir(
        cfg, "streaming", args.model, default_out_subdir(args.mode, args.noise_mask, is60)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model = models.load_model(args.model, cfg, args.device)
    logging.info("model=%s mode=%s: %d events -> %s", args.model, args.mode, len(events), out_dir)

    n_done = n_fail = n_empty = 0
    for ei, evdir in enumerate(events):
        evid = os.path.basename(evdir)
        if args.mode == "noise":
            evid = f"{os.path.basename(os.path.dirname(evdir)).split('_')[-1]}_{evid}"
        out_csv = out_dir / f"{evid}.csv"
        if out_csv.exists():
            logging.info("[%d/%d] %s: exists, skip", ei + 1, len(events), evid)
            continue
        t0 = time.time()
        try:
            if model.is_redpan60:
                df = run_event_60s(model, args.mode, evdir, args.device, noise_mask=args.noise_mask)
            else:
                df = run_event_90s(model, args.mode, evdir, args.device, args.batch_size)
        except Exception as e:
            # one unreadable event must not stop a multi-hour replay: log, count, go on
            n_fail += 1
            if n_fail == 1:
                logging.exception(
                    "[%d/%d] %s: failed (first failure; traceback follows)",
                    ei + 1,
                    len(events),
                    evid,
                )
            else:
                logging.error("[%d/%d] %s: failed: %s", ei + 1, len(events), evid, e)
            continue
        if df is None:
            logging.warning("%s: no usable stations, skipping", evid)
            n_empty += 1
            continue
        tmp = out_csv.with_name(out_csv.name + ".tmp")
        df.to_csv(tmp, index=False, sep="\t")
        os.replace(tmp, out_csv)  # atomic: a killed run never leaves a partial <EVID>.csv
        n_done += 1
        logging.info(
            "[%d/%d] %s: %d rows (%.1fs)", ei + 1, len(events), evid, len(df), time.time() - t0
        )
    n_tried = n_done + n_fail + n_empty
    if n_tried and n_done == 0:
        raise SystemExit(
            f"none of the {n_tried} events produced output ({n_fail} failed, {n_empty} without "
            f"usable stations)"
        )
    if n_fail or n_empty:
        logging.warning(
            "%d events written, %d failed, %d without usable stations", n_done, n_fail, n_empty
        )


if __name__ == "__main__":
    main()
