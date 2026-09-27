"""Template: ObsPy example waveform -> RED-PAN-Motion -> ObsPy Stream -> plot.

The smallest end-to-end demo, copy it as a starting point. No data files needed:
it uses ObsPy's bundled example earthquake (`obspy.read()` with no arguments).
Uses the redpan_motion package directly (no sys.path), the `as_stream` inference
API, and saves a figure of the input + P / S / mask / polarity prediction traces.

    PYTHONPATH=. python scripts/template_inference.py [--ckpt ...] [--out demo.png]
"""
import argparse
from pathlib import Path

import numpy as np
import obspy
import torch
from scipy.signal import butter, sosfiltfilt
# for headless runs (no display), uncomment: import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from redpan_motion import REDPANPredictor   # the model class is the only thing the package must supply

CKPT = str(Path(__file__).resolve().parents[1] / "checkpoints/redpan_motion/best.pt")


# def main():
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=CKPT)
ap.add_argument("--out", default="template_inference.png")
ap.add_argument("--mode", default="single", help="single (default) or sliding")
args = ap.parse_args()

# 1. ObsPy's bundled example: a 3-component local earthquake (BW.RJOB, 100 Hz)
st = obspy.read().detrend("demean")           # no args -> example data
dt = 1.0 / st[0].stats.sampling_rate
npts = min(len(tr.data) for tr in st)
order = {"E": 0, "1": 0, "N": 1, "2": 1, "Z": 2, "3": 2}   # last char -> E/N/Z (Z = vertical at index 2)
raw = np.zeros((3, npts), np.float32)          # (3, T) ordered E, N, Z
for tr in st:
    k = order.get(tr.stats.channel[-1].upper())
    if k is not None:
        raw[k] = tr.data[:npts]
ref = st.select(component="Z")[0]              # vertical -> timing/metadata ref
s = ref.stats
sid = f"{s.network}.{s.station}.{s.location or '--'}.{s.channel[:2]}"
print(f"{sid}   {npts * dt:.0f} s @ {1 / dt:.0f} Hz")

# 2. load the RED-PAN-Motion predictor
device = "cuda" if torch.cuda.is_available() else "cpu"
pred = REDPANPredictor.from_checkpoint(args.ckpt, device=device)

# 3. preprocess: demean + 4th-order 1 Hz highpass per channel (matches v49 training;
#    benchmarks best vs raw and the 3-45 bandpass — better picks/detection/polarity +
#    fewer noise FP). Inlined so the template is explicit (same as redpan_motion.highpass).
nyq = 0.5 / dt
sos = butter(4, 1.0 / nyq, btype="high", output="sos")
model_in = np.stack([sosfiltfilt(sos, raw[c] - raw[c].mean())
                     for c in range(3)]).astype(np.float32)

# 4. inference -> ObsPy Stream of P / S / mask / polarity traces
out = pred.predict_array(model_in, mode=args.mode, as_stream=True, reference=ref,
                         z_raw=raw[2])   # raw (unfiltered) vertical -> polarity first-motion
ch = {tr.stats.channel: tr.data[:npts] for tr in out}
print("Stream:", " ".join(tr.id for tr in out))

# 5. visualize: input vertical + each prediction trace
t = np.arange(npts) * dt
p_peak, s_peak = int(ch["P"].argmax()) * dt, int(ch["S"].argmax()) * dt
fig, ax = plt.subplots(5, 1, figsize=(11, 8), sharex=True)
z = raw[2]
ax[0].plot(t, z / (np.abs(z).max() + 1e-9), "k", lw=0.6)
ax[0].set_ylabel("Z (norm)")
ax[0].set_title(f"{sid} — RED-PAN-Motion ({args.mode}-pass)")
for a, (name, color, label) in zip(ax[1:], [("P", "C0", "P prob"),
                                            ("S", "C1", "S prob"),
                                            ("M", "C2", "mask"),
                                            ("POL", "C3", "polarity (U-D)")]):
    if name in ch:
        a.plot(t, ch[name], color, lw=1)
    a.set_ylabel(label)
for a in ax:                                   # mark the peak P (red) and S (blue)
    a.axvline(p_peak, color="r", ls="--", lw=0.8)
    a.axvline(s_peak, color="b", ls="--", lw=0.8)
for a in ax[1:4]:
    a.set_ylim(-0.05, 1.05)
ax[4].axhline(0, color="gray", lw=0.5)
ax[4].set_ylim(-1.05, 1.05)
ax[-1].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig(args.out, dpi=120)
print(f"P peak {p_peak:.2f} s · S peak {s_peak:.2f} s   ->  saved {args.out}")


# if __name__ == "__main__":
#     main()
