"""CLI: single-station inference -> redpan_picks CSV.

Thin wrapper over the redpan_motion package (no sys.path hacks). The real logic
lives in `redpan_motion.waveform_io` (load / bandpass) and `redpan_motion.picks`
(mask-detection pairing + DataFrame). For batch/archive runs use
`scripts/daily_inference.py`; for interactive use see `notebooks/daily_inference.py`.

    python scripts/picks_to_csv.py --data 'day/*.sac' [--xml STA.xml] [--mode single|sliding]
"""
import argparse
from pathlib import Path

import torch

from redpan_motion import (
    REDPANPredictor, load, highpass, picks_to_dataframe, sensor_type, CH_Z,
)

ROOT = str(Path(__file__).resolve().parent.parent)   # project root (this file is scripts/..)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="glob matching one instrument's three components, e.g. 'day/*.sac'")
    ap.add_argument("--xml", default=None,
                    help="StationXML for the station; needed for amplitude columns")
    ap.add_argument("--ckpt", default=f"{ROOT}/checkpoints/redpan_motion/best.pt")
    ap.add_argument("--out", default=f"{ROOT}/outputs/redpan_picks")
    ap.add_argument("--mode", default="single", help="single (default) or sliding")
    ap.add_argument("--no-amplitude", action="store_true",
                    help="fast path: skip per-pick response removal "
                         "(amp/WA/snr columns blank, no StationXML needed)")
    ap.add_argument("--two-pass-polarity", action="store_true",
                    help="run a second raw-input forward for polarity (~2x inference). "
                         "Default: single pass (band-pass picks + raw-Z polarity in one forward).")
    args = ap.parse_args()

    amp = not args.no_amplitude
    if amp and args.xml is None:
        print("no --xml given: amplitude columns need station response, so they are skipped")
        amp = False
    raw, wf_so, inv, (net, sta, loc, chn_pre), t0, st = load(
        args.data, args.xml if amp else None)
    dt = 1.0 / st[0].stats.sampling_rate
    station_id = f"{net}.{sta}.{loc}.{chn_pre}"
    print(f"{station_id}  {raw.shape[1] / 100 / 3600:.1f} h  mode={args.mode}  "
          f"amplitude={'on' if amp else 'off'}"
          f"{'  sensor=' + sensor_type(station_id) if amp else ''}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = REDPANPredictor.from_checkpoint(args.ckpt, device=device)
    if args.two_pass_polarity:
        # pass 1 highpass -> picks/detection; pass 2 raw -> polarity (the older
        # two-pass recipe; the single pass below is the recommended one)
        picker, detector, _ = pred.predict_arrays(highpass(raw, 1.0, dt), mode=args.mode)
        _, _, polarity = pred.predict_arrays(raw.astype("float32"), mode=args.mode)
    else:
        # single forward: 1 Hz highpass picks/detection + raw-Z polarity together,
        # one read and one pass
        picker, detector, polarity = pred.predict_arrays(
            highpass(raw, 1.0, dt), mode=args.mode, postprocess=False,
            z_raw=raw[CH_Z])   # raw (unfiltered) vertical -> polarity first-motion
    df = picks_to_dataframe(picker, detector, polarity, t0, station_id,
                            raw_counts=raw, wf_sensonly=wf_so, inv=inv, dt=dt, amplitude=amp)

    out_dir = Path(args.out) / f"{t0.year:04d}" / f"{t0.julday:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"picks_{station_id}.csv"
    df.to_csv(out_file, index=False)
    print(f"wrote {len(df)} pick rows ({len(df) // 2} events) -> {out_file}")
    print(df.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
