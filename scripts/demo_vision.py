"""Demo: run the validation battery, save figures and a report."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from hvp import battery as bat
from hvp import baseline as B
from hvp.pipeline import VisionPipeline
from hvp import stimuli as S

OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)


def fig_flicker(res):
    freqs = sorted(res["variances"])
    vals = [res["variances"][f] for f in freqs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(freqs, vals, "o-", color="#7b2ff7", lw=2, ms=8)
    ax.axvline(B.FLICKER_FUSION_HZ, ls="--", color="gray",
               label=f"human fusion ~{B.FLICKER_FUSION_HZ:.0f} Hz")
    ax.set_xlabel("flicker frequency (Hz)")
    ax.set_ylabel("percept-stream temporal variance")
    ax.set_title("T1 — flicker fusion: variance collapses by 60 Hz")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "t1_flicker.png"), dpi=110)
    plt.close(fig)


def fig_wheel(res):
    frames = res["frames_aliased"]
    idx = np.linspace(0, len(frames) - 1, 6).astype(int)
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.6))
    for ax, i in zip(axes, idx):
        ax.imshow(frames[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={(i + 2) * 100} ms")
        ax.axis("off")
    fig.suptitle(f"T2 — wagon-wheel: 9 Hz spoke-pass reads "
                 f"{res['aliased_9Hz']:+.3f} rev/s (reversed); "
                 f"2 Hz control {res['control_2Hz']:+.3f} rev/s (veridical)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "t2_wheel.png"), dpi=110)
    plt.close(fig)


def fig_suppression():
    from hvp.battery import _run, _dva_per_px
    size, fps = 64, 1000
    cx = size / 2
    dva = _dva_per_px(size)
    amp_px = 13.0 / dva
    gen = lambda: S.gen_flash(1.6, fps, size, flash_at_ms=1000.0,
                              flash_dur_ms=20.0)
    m_fix = _run([(0, cx, cx)], gen(), size)
    m_sac = _run([(0, cx, cx), (990.0, cx + amp_px, cx)], gen(), size)
    t = [m[0] for m in m_fix]
    y_fix = [m[1].mean() - 0.3 for m in m_fix]
    y_sac = [m[1].mean() - 0.3 for m in m_sac]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, y_fix, "o-", label="flash during fixation", color="#7b2ff7")
    ax.plot(t, y_sac, "s-", label="flash mid-saccade", color="#ff5c5c")
    ax.axvspan(1000, 1020, color="gold", alpha=0.25, label="flash (20 ms)")
    ax.axvspan(990, 1040, color="gray", alpha=0.15, label="saccade window")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("percept deviation from baseline")
    ax.set_title("T3 — saccadic suppression: the mid-saccade flash is not seen")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "t3_suppression.png"), dpi=110)
    plt.close(fig)


def fig_foveation():
    # synthetic scene: fine detail everywhere, fixate center
    size = 256
    img = Image.new("L", (size, size), 128)
    d = ImageDraw.Draw(img)
    for y in range(0, size, 16):
        d.line([0, y, size, y], fill=200, width=1)
    for x in range(0, size, 16):
        d.line([x, 0, x, size], fill=200, width=1)
    for i in range(0, size, 32):
        d.text((i + 2, i + 2), "HVP", fill=255)
    scene = np.asarray(img, dtype=np.float32) / 255.0
    dva = B.FIELD_WIDTH_DEG / size
    pipe = VisionPipeline((size, size), dva, [(0, size / 2, size / 2)])
    for i in range(300):  # 300 ms of static scene at 1000 fps
        pipe.push(scene, i * 1.0)
    moments = list(pipe.moments(300.0))
    retinal = moments[-1][1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].imshow(scene, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("photon stream (camera view)")
    axes[0].axis("off")
    axes[1].imshow(retinal, cmap="gray", vmin=0, vmax=1)
    axes[1].plot(size / 2, size / 2, "r+", ms=14, mew=2)
    axes[1].set_title("retinal view (foveated, fixated center)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "foveation.png"), dpi=110)
    plt.close(fig)


def fig_change_blindness(res):
    from hvp import attention as A
    cyc_no = np.array(res["cycles_no_mask"])
    cyc_m = np.array(res["cycles_mask"])
    ex = res["example_masked_run"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # left: cycles-to-detection per seed
    ax = axes[0]
    rng = np.random.default_rng(7)
    for i, (c, color, label) in enumerate(
            [(cyc_no, "#7b2ff7", "no mask"), (cyc_m, "#ff5c5c", "mask")]):
        x = rng.normal(i, 0.06, size=len(c))
        censored = c >= 25.0
        ax.scatter(x[~censored], c[~censored], s=70, color=color, zorder=3)
        ax.scatter(x[censored], c[censored], s=110, color=color,
                   marker="^", zorder=3)  # censored: never found in 25
        ax.scatter([i], [np.mean(c)], s=160, color=color, marker="D",
                   edgecolor="black", zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no mask", "mask"])
    ax.set_ylabel("paradigm cycles to detection")
    ax.set_title(f"T4 — change blindness: no-mask "
                 f"{np.mean(cyc_no):.1f} vs masked {np.mean(cyc_m):.1f} "
                 f"cycles\n(triangles: never found in 25; diamonds: means)")
    ax.set_ylim(0, 28)

    # right: example masked scanpath (never found the change)
    ax = axes[1]
    ax.imshow(ex["scene_small"], cmap="gray", vmin=0, vmax=1)
    sp = np.array(ex["scanpath"]) / A.SCALE
    ax.plot(sp[:, 1], sp[:, 2], "o-", color="#ffb020", ms=4, lw=1,
            alpha=0.8)
    ax.plot(sp[0, 1], sp[0, 2], "go", ms=8, label="start")
    chx, chy = ex["change_xy_small"]
    circ = plt.Circle((chx, chy), 3.5, color="red", fill=False, lw=2,
                      label="the change (never foveated)")
    ax.add_patch(circ)
    ax.set_title("masked search path: 25 cycles, change never foveated")
    ax.legend(fontsize=8, loc="lower right")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "t4_change_blindness.png"), dpi=110)
    plt.close(fig)


def main():
    print("running validation battery...", flush=True)
    results = bat.run_all()
    for name, r in results.items():
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[{status}] {name}: {r['detail']}", flush=True)

    print("rendering figures...", flush=True)
    fig_flicker(results["T1_flicker"])
    fig_wheel(results["T2_wagon_wheel"])
    fig_suppression()
    fig_change_blindness(results["T4_change_blindness"])
    fig_foveation()

    lines = ["# HVP validation report", ""]
    for name, r in results.items():
        lines.append(f"## {name} — {'PASS' if r['pass'] else 'FAIL'}")
        lines.append(r["detail"])
        lines.append("")
    lines.append("Figures: t1_flicker.png, t2_wheel.png, t3_suppression.png, "
                 "t4_change_blindness.png, foveation.png")
    with open(os.path.join(OUT, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("done. output/REPORT.md + figures written.", flush=True)
    return all(r["pass"] for r in results.values())


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
