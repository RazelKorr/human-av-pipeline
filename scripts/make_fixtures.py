"""make_fixtures.py: deterministic generators for synthetic fixtures.

Fixtures that can be generated belong in code, not in the media
folder's oral history: anyone with ffmpeg can rebuild them
bit-comparably and the regime tests in
tests/test_fixture_regimes.py can state exactly what the fixture
contains (see tests/fixtures/FIXTURES.md for the corpus table).

Currently generated:

  quiet_room_60s.mp4 -- the quiet/near-silent regime (audit item
  11's missing fixture). 60 s, 640x360 @ 30 fps. Visual: near-
  black base (0x050505, luma ~0.012 decoded, below hvp's
  DARK_LUM 0.03)
  with ONE slow faint brightening (gaussian in time, +0.045 luma
  peak, centered t = 42 s, sigma 5 s). Audio: seeded brown-noise
  floor at ~-66 dBFS RMS (below the streaming ear's 1e-3 RMS
  gate) with exactly ONE planted event -- an 880 Hz
  chime, 0.6 s with 50 ms fades (no click transients), ~-30 dBFS
  peak, at t = 30.0 s. The chime and the brightening are 12 s
  apart on purpose: nothing in this fixture should AV-bind.

Usage:
  python3 scripts/make_fixtures.py            # writes input/quiet_room_60s.mp4
  python3 scripts/make_fixtures.py --out /tmp/quiet.mp4
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Fixture constants -- the tests and FIXTURES.md cite these.
# Levels are CALIBRATED TO MEASURED OUTPUT (first render, 2026-10-12):
# lavfi's sine source peaks at ~0.15 (not 1.0) and anoisesrc brown
# amplitude is not RMS, so the constants below are the encoder
# inputs that produce the documented levels: base luma ~0.012
# decoded, noise floor RMS ~0.0005 (-66 dBFS, below the streaming
# ear's 1e-3 RMS gate), chime peak ~0.03 (-30 dBFS).
QUIET_DURATION_S = 60.0
QUIET_BASE_HEX = "0x050505"
QUIET_BASE_LUMA = 5 / 255.0            # pre-encode; ~0.012 decoded
QUIET_LIGHT_T_S = 42.0                 # brightening center
QUIET_LIGHT_SIGMA_S = 5.0
QUIET_LIGHT_PEAK = 0.045               # added luma at center
QUIET_NOISE_AMP = 0.0025               # -> RMS ~0.0005 measured
QUIET_NOISE_SEED = 1234
QUIET_CHIME_T_S = 30.0
QUIET_CHIME_FREQ_HZ = 880.0
QUIET_CHIME_DUR_S = 0.6
QUIET_CHIME_VOLUME = 0.20              # sine peak 0.15 x 0.20 -> ~0.03


def build_quiet_room(out_path):
    """Render the quiet-room fixture with ffmpeg lavfi sources.

    Deterministic: fixed lavfi sources, fixed noise seed, fixed
    encoder settings. Returns the ffmpeg command used.
    """
    bright = (f"{QUIET_LIGHT_PEAK}*exp(-pow((t-{QUIET_LIGHT_T_S:.0f})"
              f"/{QUIET_LIGHT_SIGMA_S:.0f}\\,2))")
    vfilter = (f"color=c={QUIET_BASE_HEX}:s=640x360:r=30"
               f":d={QUIET_DURATION_S:.0f},"
               f"eq=brightness='{bright}':eval=frame,format=yuv420p")
    noise = (f"anoisesrc=color=brown:amplitude={QUIET_NOISE_AMP}"
             f":duration={QUIET_DURATION_S:.0f}:sample_rate=44100"
             f":seed={QUIET_NOISE_SEED}")
    chime = (f"sine=frequency={QUIET_CHIME_FREQ_HZ:.0f}"
             f":duration={QUIET_CHIME_DUR_S}:sample_rate=44100,"
             f"afade=t=in:st=0:d=0.05,"
             f"afade=t=out:st={QUIET_CHIME_DUR_S - 0.05:.2f}:d=0.05,"
             f"volume={QUIET_CHIME_VOLUME},"
             f"adelay={int(QUIET_CHIME_T_S * 1000)}:all=1")
    cmd = ["ffmpeg", "-y",
           "-f", "lavfi", "-i", vfilter,
           "-f", "lavfi", "-i", noise,
           "-f", "lavfi", "-i", chime,
           "-filter_complex",
           "[1:a][2:a]amix=inputs=2:duration=first:normalize=0[a]",
           "-map", "0:v", "-map", "[a]",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-c:a", "aac", "-shortest", out_path]
    subprocess.run(cmd, check=True)
    return cmd


CROSSED_DURATION_S = 60.0


def build_dark_busy(out_path):
    """Dark-but-busy crossed fixture (2026-10-13, EXPERIMENTS).

    60 s, 640x360 @ 30 fps. Visual: near-black base (RGB 2) with
    TWO mid-grey squares in continuous motion (a 70x70 square
    sweeping horizontally, a 40x40 square sweeping vertically)
    -- measured mean luma ~0.02, below DARK_LUM 0.03, while
    local motion is present throughout (busy regime). Audio:
    brown-noise bed at a clearly audible level plus 12 planted
    tone bursts (880/660 Hz alternating, 0.3 s, 50 ms fades) at
    t = 2.5 + 5k s. Ground truth by construction: motion present
    for the full minute; 12 planted audio events.
    """
    # Frames are rendered in numpy and piped as rawvideo: the
    # drawbox-expression version of this fixture rendered the
    # squares STATIC (measured motion proxy 0.0, 2026-10-13), so
    # the moving ground truth is computed per-frame here instead.
    noise = (f"anoisesrc=color=brown:amplitude=0.05"
             f":duration={CROSSED_DURATION_S:.0f}:sample_rate=44100"
             f":seed={QUIET_NOISE_SEED}")
    bursts = []
    for k in range(12):
        freq = 880.0 if k % 2 == 0 else 660.0
        bursts.append(
            f"sine=frequency={freq:.0f}:duration=0.3:sample_rate=44100,"
            "afade=t=in:st=0:d=0.05,afade=t=out:st=0.25:d=0.05,"
            f"volume=0.6,adelay={int((2.5 + 5 * k) * 1000)}:all=1")
    inputs = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x360",
              "-r", "30", "-i", "-"]
    for src in [noise] + bursts:
        inputs += ["-f", "lavfi", "-i", src]
    n_audio = 1 + len(bursts)
    mix_inputs = "".join(f"[{i}:a]" for i in range(1, n_audio + 1))
    cmd = (["ffmpeg", "-y"] + inputs + [
        "-filter_complex",
        f"{mix_inputs}amix=inputs={n_audio}:duration=first:normalize=0[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", out_path])
    import math
    import numpy as _np
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    base = _np.full((360, 640, 3), 2, dtype=_np.uint8)
    for i in range(int(CROSSED_DURATION_S * 30)):
        t = i / 30.0
        fr = base.copy()
        x1 = int((t * 240.0) % 570.0)
        y1 = int(140 + 80 * math.sin(3.0 * t))
        fr[y1:y1 + 70, x1:x1 + 70] = (119, 119, 119)
        x2 = int(300 + 120 * math.sin(2.0 * t))
        y2 = int((t * 180.0) % 320.0)
        fr[y2:y2 + 40, x2:x2 + 40] = (140, 140, 140)
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg dark_busy failed: {proc.returncode}")
    return cmd


def build_bright_silent(out_path):
    """Bright-but-silent crossed fixture (2026-10-13, EXPERIMENTS).

    60 s, 640x360 @ 30 fps. Visual: solid bright field
    (0xC8C8C8, luma ~0.78), completely static. Audio: digital
    silence (anullsrc). Ground truth by construction: zero audio
    events, zero visual motion/transients; the darkness gate has
    nothing to engage on, the silence/dwell side has everything.
    """
    cmd = ["ffmpeg", "-y",
           "-f", "lavfi", "-i",
           (f"color=c=0xC8C8C8:s=640x360:r=30"
            f":d={CROSSED_DURATION_S:.0f},format=yuv420p"),
           "-f", "lavfi", "-i",
           f"anullsrc=channel_layout=mono:sample_rate=44100",
           "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-c:a", "aac", "-t", f"{CROSSED_DURATION_S:.0f}", out_path]
    subprocess.run(cmd, check=True)
    return cmd


BUILDERS = {"quiet": build_quiet_room, "dark_busy": build_dark_busy,
            "bright_silent": build_bright_silent}
DEFAULT_OUTS = {"quiet": "quiet_room_60s.mp4",
                "dark_busy": "dark_busy_60s.mp4",
                "bright_silent": "bright_silent_60s.mp4"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=sorted(BUILDERS), default="quiet")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "input",
                                   DEFAULT_OUTS[args.kind])
    BUILDERS[args.kind](out)
    args.out = out
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of", "csv=p=0", args.out],
        capture_output=True, text=True)
    print(f"wrote {args.out} (duration {probe.stdout.strip()} s)")


if __name__ == "__main__":
    sys.exit(main())
