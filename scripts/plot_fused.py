"""Plot the fused audio-visual feed: gaze + audio attention on one timeline."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FUSED = os.path.join(ROOT, "output", "fused")

recs = [json.loads(l) for l in open(os.path.join(FUSED, "av_joint.jsonl"))]
t = np.array([r["t_s"] for r in recs])
gx = np.array([r["gaze_x"] for r in recs])
gy = np.array([r["gaze_y"] for r in recs])
cf = np.array([r["audio_cf_hz"] for r in recs])
aev = np.array([r["audio_event"] for r in recs])

fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

axes[0].plot(t, gx, c="cyan", lw=0.8, label="gaze x")
axes[0].plot(t, gy, c="magenta", lw=0.8, label="gaze y")
axes[0].set_ylabel("gaze (224px)")
axes[0].set_title("Star Tours 0-62s: fused audio-visual feed (10 Hz, media time)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_ylim(-10, 234)

axes[1].plot(t, cf, c="white", lw=0.8)
axes[1].set_yscale("log")
axes[1].set_ylabel("audio cf (Hz, log)")
axes[1].set_ylim(40, 9000)

# joint events: saccades (cyan) + audio events (red)
sac_t = [r["t_s"] for r in recs if r["saccades"]]
for tt in sac_t:
    axes[2].axvline(tt, color="cyan", alpha=0.25, lw=0.6)
for tt, e in zip(t, aev):
    if e in ("onset-capture", "switch"):
        axes[2].axvline(tt, color="red", alpha=0.35, lw=0.6)
axes[2].set_xlim(0, 62)
axes[2].set_xlabel("time (s)")
axes[2].set_yticks([])
axes[2].set_title("events: saccades (cyan) vs audio switch/capture (red)",
                  fontsize=10)

for ax in axes:
    ax.set_facecolor("black")
    ax.grid(True, alpha=0.15)
fig.patch.set_facecolor("#111111")
fig.tight_layout()
out = os.path.join(FUSED, "av_fused.png")
fig.savefig(out, dpi=110, facecolor=fig.get_facecolor())
print("wrote", out, f"({len(sac_t)} joint moments with saccades)")
