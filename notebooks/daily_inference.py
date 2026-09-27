# %% [markdown]
# # RED-PAN-Motion daily inference — interactive
#
# Open this file in Jupyter (via jupytext) or VS Code and run it cell by cell.
# No `sys.path` manipulation: after `pip install -e .` (run once from the repo
# root) every `redpan_motion` symbol imports directly.
#
# ```bash
# pip install -e /path/to/RED-PAN-Motion
# ```

# %%
import os
from pathlib import Path

import pandas as pd
import torch

from redpan_motion import (
    REDPANPredictor,
    load,                 # read 3-comp SAC (+ optional StationXML)
    highpass,             # demean + 1 Hz highpass -> the recommended model input
    picks_to_dataframe,   # model output -> redpan_picks DataFrame
    group_instruments,    # day archive -> per-instrument glob indices
)

# Repo root: REDPAN_MOTION_ROOT env var, else the directory above notebooks/.
try:
    _default_root = str(Path(__file__).resolve().parents[1])
except NameError:  # running as Jupyter cells (no __file__); assume cwd is notebooks/
    _default_root = str(Path.cwd().parent)
ROOT = os.environ.get("REDPAN_MOTION_ROOT", _default_root)
CKPT = f"{ROOT}/checkpoints/redpan_motion/best.pt"
# Root of your day archive: <DATADIR>/<NETDIR>/A/<YYYY>/<JJJ>/<STA>/<files>
DATADIR = os.environ.get("REDPAN_DATADIR", "/path/to/day_archive")

# %% [markdown]
# ## Load the model once

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
pred = REDPANPredictor.from_checkpoint(CKPT, device=device)
print("loaded", CKPT, "on", device)

# %% [markdown]
# ## One instrument: load -> highpass -> predict_arrays -> picks
# `predict_arrays` returns the polarity head explicitly as the 3rd element.
# The paths below follow a CWA-style day archive, which is not public: point
# them at your own three-component files.

# %%
gidx = f"{DATADIR}/NETDIR/A/2019/001/STA/STA.NET.LOC.HH?.2019.001"   # one instrument, ? = component
raw, wf_so, inv, (net, sta, loc, chn2), t0, st = load(gidx, xml=None)   # xml=None -> no-amplitude
dt = 1.0 / st[0].stats.sampling_rate
station_id = f"{net}.{sta}.{loc}.{chn2}"

picker, detector, polarity = pred.predict_arrays(highpass(raw, 1.0, dt), mode="single")
# picker (T,3) [P,S,Noise] · detector (T,2) [mask,unmask] · polarity (T,3) [N,U,D]

df = picks_to_dataframe(picker, detector, polarity, t0, station_id,
                        raw_counts=raw, wf_sensonly=wf_so, inv=inv, dt=dt, amplitude=False)
print(station_id, "->", len(df) // 2, "P-S pairs")
df.head(10)

# %% [markdown]
# ## A whole day: iterate every instrument in the archive

# %%
groups = group_instruments(DATADIR, year=2019, jday=1)
print(len(groups), "3-component instruments")

all_picks = []
for (net, sta, loc, chn2), gidx in groups.items():
    try:
        raw, wf_so, inv, ids, t0, st = load(gidx, xml=None)
        dt = 1.0 / st[0].stats.sampling_rate
        picker, detector, polarity = pred.predict_arrays(highpass(raw, 1.0, dt), mode="single")
        df = picks_to_dataframe(picker, detector, polarity, t0, ".".join(ids), dt=dt, amplitude=False)
    except Exception as e:                          # keep going if one instrument is unreadable
        print(f"  {net}.{sta}.{loc}.{chn2}: SKIP ({type(e).__name__}: {e})")
        continue
    all_picks.append(df)
    print(f"  {'.'.join(ids)}: {len(df) // 2} pairs")

# %%
day = pd.concat(all_picks, ignore_index=True)
day.groupby("type").size()
