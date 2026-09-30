"""Visualize the Star Tours audio pipeline: cochleagram, salience, attention trace."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hva import cochlea as C

INDIR = os.path.join(os.path.dirname(__file__), "..", "output",
                     "star_tours_62s", "dwell")

Sg = np.load(os.path.join(INDIR, "cochleagram.npy"))
sal = np.load(os.path.join(INDIR, "salience.npy"))
f = np.load(os.path.join(INDIR, "freqs_hz.npy"))
t = np.load(os.path.join(INDIR, "times_s.npy"))

trace = []
for line in open(os.path.join(INDIR, "trace.txt")).readlines()[1:]:
    p = line.split()
    trace.append((float(p[0]), float(p[1]), p[4] if len(p) > 4 else None))
tt = np.array([x[0] for x in trace])
cf = np.array([x[1] for x in trace])
cf_hz = C.bin_to_hz(np.clip(np.round(cf).astype(int), 0, 63))
events = [(x[0], x[2]) for x in trace if x[2] not in (None, "None")]

# downsample spectrograms for display (every 4th frame)
ds = 4
Sg_d = Sg[::ds]
sal_d = sal[::ds]
t_d = t[::ds]

fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# cochleagram
im0 = ax[0].imshow(Sg_d.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[t_d[0], t_d[-1], 0, 63], vmin=-80, vmax=-10)
ax[0].set_ylabel("freq bin (0=50Hz, 63=8kHz)")
ax[0].set_title("Star Tours 0-62s: cochleagram (dBFS)")
plt.colorbar(im0, ax=ax[0], label="dBFS")

# salience
im1 = ax[1].imshow(sal_d.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[t_d[0], t_d[-1], 0, 63], vmin=0, vmax=sal.max())
ax[1].set_ylabel("freq bin")
ax[1].set_title("Salience map")
plt.colorbar(im1, ax=ax[1], label="salience")

# attention trace
ax[2].plot(tt, cf_hz, "w-", lw=0.8, alpha=0.9)
ax[2].set_yscale("log")
ax[2].set_ylabel("attended freq (Hz, log)")
ax[2].set_xlabel("time (s)")
ax[2].set_title("Attention trace (center frequency)")
ax[2].set_ylim(40, 9000)
for et, ev in events:
    if ev == "onset-capture":
        ax[2].axvline(et, color="red", alpha=0.3, lw=0.8)
    elif ev == "switch":
        ax[2].axvline(et, color="cyan", alpha=0.2, lw=0.8)
ax[2].set_facecolor("black")
ax[2].grid(True, alpha=0.2)

plt.tight_layout()
out = os.path.join(INDIR, "overview.png")
plt.savefig(out, dpi=100)
print("wrote", out)
