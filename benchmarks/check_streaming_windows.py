#!/usr/bin/env python
"""Short-window check for the 60 s streaming replay (``run_streaming.py``, 60 s RED-PAN models).

The 60 s replay slices each window as ``stream.slice(end - 60 s, end)`` and passes it to
``stream_standardize``, which truncates to 6,000 samples or, when a component returns fewer samples,
inserts zeros before its LAST sample (``np.insert(x, -1, zeros)``). A window that starts before a
component's first sample is therefore left-aligned at the record start while the pick time is still
computed as ``window_start + index * 0.01 s``, which moves the pick earlier and the delay later by the
missing lead. This script reproduces the replay's reading, filtering and windowing, flags every
window whose slice is shorter than 6,000 samples on any component (with its lead and tail gaps),
and scores the replay (``score_streaming.score``) with and without those windows:

  * number of EQ / noise windows that are short (and how many are time-shifted by a lead gap);
  * how many fire-ASAP detections (the earliest qualifying window per station record at the
    fire-ASAP operating point) and how many fire-ASAP false positives come from short windows;
  * best-F1 and fire-ASAP scores on all rows, with short-window rows dropped, and with every
    station record that has a short window excluded.

The 90 s replay places each trace by its start-time offset (``pad_and_zscore``), so it has no
such shift and is not covered here.

Example:
    python check_streaming_windows.py --eq-dir results/streaming/redpan/eq \\
        --noise-dir results/streaming_mask/redpan/noise --out-json results/redpan_short_windows.json
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob

import numpy as np
import pandas as pd
from obspy import read

from rpm_bench import config
from rpm_bench.streaming_utils import list_eq_events, list_noise_subdirs, sac_len_complement
from score_streaming import DMAX, epoch_seconds, gather, score

PRED_SEC, INTERVAL, DT, NPTS = 60, 0.05, 0.01, 6000


def _slice_npts(tr, stt, ent):
    """Samples ``tr.slice(stt, ent)`` returns (nearest-sample trim, no padding)."""
    return int(round((min(tr.stats.endtime, ent) - max(tr.stats.starttime, stt)) / DT)) + 1


def _window_rows(wf, wf_ent, key):
    rows = []
    for e in wf_ent:
        s = e - PRED_SEC
        rows.append(
            dict(
                **key,
                inference_endtime=str(e)[:22],
                nmin=min(_slice_npts(t, s, e) for t in wf),
                lead=max(max(0.0, float(t.stats.starttime - s)) for t in wf),
                tail=max(max(0.0, float(e - t.stats.endtime)) for t in wf),
            )
        )
    return rows


def eq_windows(eq_root):
    """Every window of the EQ replay, as ``run_event_60s`` builds it."""
    rows = []
    for evdir in list_eq_events(eq_root):
        evid = os.path.basename(evdir)
        net, ev = evid.split("_")[0], evid.split("_")[1]
        for sta in np.unique(
            [os.path.basename(i)[:-5] for i in glob(os.path.join(evdir, "*.sac"))]
        ):
            ri = os.path.join(evdir, f"{sta}*.sac")
            wf = read(ri)
            wf = sac_len_complement(wf.resample(100) if wf[0].stats.sampling_rate != 100 else wf)
            wf.sort()
            info = wf[0].stats
            p_utc = info.starttime + info.sac.t1 - info.sac.b
            if int((p_utc - info.starttime) / DT) < 100 or (info.endtime - p_utc < 5):
                continue
            n = int(np.round(6 / INTERVAL))
            wf_ent = np.sort([p_utc - 1 + INTERVAL * N for N in range(n)])
            key = dict(ID=f"{net}_{wf[0].id[:-1]}_{ev}", p_offset=float(p_utc - info.starttime))
            rows += _window_rows(wf, wf_ent, key)
    return pd.DataFrame(rows)


def noise_windows(noise_root):
    rows = []
    for evdir in list_noise_subdirs(noise_root):
        net = os.path.split(evdir)[0].split("_")[-1]
        ev = os.path.basename(evdir)
        for sta in np.unique(
            [os.path.basename(i).split(".")[1] for i in glob(os.path.join(evdir, "*.sac"))]
        ):
            wf = read(os.path.join(evdir, f"*.{sta}.*.sac"), headonly=True)
            wf.sort()
            start_utc = wf[0].stats.starttime + PRED_SEC
            n = int(np.round((wf[0].stats.endtime - start_utc) / INTERVAL))
            wf_ent = np.sort([start_utc + INTERVAL * N for N in range(n)])
            rows += _window_rows(wf, wf_ent, dict(ID=f"{net}_{wf[0].id[:-1]}_{ev}"))
    return pd.DataFrame(rows)


def flag(df, win):
    w = win[win.nmin < NPTS][["ID", "inference_endtime", "lead"]]
    out = df.merge(w, on=["ID", "inference_endtime"], how="left")
    out["short"] = out["lead"].notna()
    return out


def asap_rows(eq, nz, op):
    """Fire-ASAP detections (earliest qualifying window per record) and FP rows at operating point ``op``."""
    pp, mp, dmin = op

    def sel(d):
        return d[
            (d.delay >= 0)
            & (d.delay <= DMAX)
            & (d.P_probability >= pp)
            & (d.Mask_probability >= mp)
            & (d.delay >= dmin)
        ]

    fe = sel(eq)
    det = fe.loc[fe.groupby("ID")["delay"].idxmin()] if len(fe) else fe
    fnz = sel(nz).copy()
    fnz["tt"] = (epoch_seconds(pd.to_datetime(fnz["inference_endtime"])) - fnz.delay).round(2)
    return det, fnz


def fmt(x):
    return dict(
        f1=x[0],
        recall=x[1],
        precision=x[2],
        mean_delay=x[3],
        tp=x[4],
        fp=x[5],
        P=x[6],
        M=x[7],
        dmin=x[8],
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument("--eq-dir", required=True, help="EQ replay CSVs of a 60 s model")
    p.add_argument("--noise-dir", required=True, help="noise replay CSVs of the same model")
    p.add_argument(
        "--windows-csv", default=None, help="also write the per-window flags of short windows"
    )
    p.add_argument("--out-json", default=None)
    args = p.parse_args()
    cfg = config.load(args.config)
    root = cfg["eew_root"]

    ew, nw = eq_windows(str(root / "EQ_Mw4_to_6.9")), noise_windows(str(root / "noise"))
    eq, nz = flag(gather(args.eq_dir), ew), flag(gather(args.noise_dir), nw)
    rec_short = set(ew.loc[ew.nmin < NPTS, "ID"])
    res = {
        "windows": {
            "eq": dict(
                n=len(ew),
                short=int((ew.nmin < NPTS).sum()),
                shifted=int(((ew.nmin < NPTS) & (ew.lead > DT)).sum()),
                records=int(ew.ID.nunique()),
                records_with_short=len(rec_short),
                min_p_offset_s=float(ew.p_offset.min()),
                min_first_window_margin_s=float(ew.p_offset.min() - 61),
            ),
            "noise": dict(
                n=len(nw),
                short=int((nw.nmin < NPTS).sum()),
                shifted=int(((nw.nmin < NPTS) & (nw.lead > DT)).sum()),
                records=int(nw.ID.nunique()),
                records_with_short=int(nw.loc[nw.nmin < NPTS, "ID"].nunique()),
            ),
        }
    }
    if args.windows_csv:
        pd.concat(
            [ew[ew.nmin < NPTS].assign(kind="eq"), nw[nw.nmin < NPTS].assign(kind="noise")]
        ).to_csv(args.windows_csv, index=False)

    total, best, a = score(eq, nz)
    det, fps = asap_rows(eq, nz, a[6:9])
    res["all"] = dict(
        n_eq=total,
        best=fmt(best),
        asap=fmt(a),
        asap_detections_from_short=int(det.short.sum()),
        asap_fp_rows_from_short=int(fps.short.sum()),
        asap_fp_triggers_from_short=int(
            fps[fps.short].groupby(["network", "station_id", "evid", "tt"]).ngroups
        ),
    )
    t2, b2, a2 = score(eq[~eq.short], nz[~nz.short])
    res["short_rows_dropped"] = dict(n_eq=t2, best=fmt(b2), asap=fmt(a2))
    t3, b3, a3 = score(eq[~eq.ID.isin(rec_short)], nz[~nz.short])
    res["short_records_excluded"] = dict(n_eq=t3, best=fmt(b3), asap=fmt(a3))

    w = res["windows"]
    print(
        f"EQ windows: {w['eq']['n']:,}, short {w['eq']['short']} (time-shifted {w['eq']['shifted']}); "
        f"records {w['eq']['records']}, with a short window {w['eq']['records_with_short']}; "
        f"P is >= {w['eq']['min_p_offset_s']:.2f} s after record start (first window starts "
        f">= {w['eq']['min_first_window_margin_s']:.2f} s inside the record)"
    )
    print(
        f"noise windows: {w['noise']['n']:,}, short {w['noise']['short']} (time-shifted {w['noise']['shifted']}) "
        f"in {w['noise']['records_with_short']} of {w['noise']['records']} records"
    )
    print(
        f"fire-ASAP detections from short windows: {res['all']['asap_detections_from_short']} of {a[4]}; "
        f"fire-ASAP FP triggers from short windows: {res['all']['asap_fp_triggers_from_short']} of {a[5]}"
    )
    for lab in ("all", "short_rows_dropped", "short_records_excluded"):
        r = res[lab]
        for k in ("best", "asap"):
            x = r[k]
            print(
                f"  {lab:22s} {k:5s} F1={x['f1']:.4f} R={x['recall']:.4f} P={x['precision']:.4f} "
                f"delay={x['mean_delay']:.3f}s FP={x['fp']} @P{x['P']}/M{x['M']}/d{x['dmin']} (n_eq={r['n_eq']})"
            )
    if args.out_json:
        json.dump(res, open(args.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
