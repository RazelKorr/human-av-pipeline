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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        ROOT, "input", "quiet_room_60s.mp4"))
    args = ap.parse_args()
    build_quiet_room(args.out)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of", "csv=p=0", args.out],
        capture_output=True, text=True)
    print(f"wrote {args.out} (duration {probe.stdout.strip()} s)")


if __name__ == "__main__":
    sys.exit(main())
