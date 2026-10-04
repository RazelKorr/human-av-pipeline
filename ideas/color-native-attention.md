# Color-native attention driver

Filed 2026-10-01, from Mykal: "colors can drive attention just like brightness."
**Built 2026-10-02** (overnight commission): `hvp/attention.chroma_salience()`
+ `salience_map(..., small_rgb=..., chroma_weight=...)`; video driver
(`scripts/run_video_saccades_color.py`) pass 1 now streams color 224px
frames and adds the chroma channel. Grayscale callers (T4 trials) are
byte-identical — the channel is exactly 0 without `small_rgb`.

The attention driver (`hvp/attention.py`) is still grayscale-native: it builds
salience from luminance center-surround + a transient (change) channel. The
render path went color in 2026-10-01 (foveated color: sharp luma, color-blind
periphery), but saccade *targets* are still picked without any chroma
information.

Why it matters: color is a genuine attentional cue in human vision — a red
warning light, a green Endor canopy against black space, the blue flash of a
lightsaber. A luminance-only driver systematically under-predicts fixations
on chromatically salient targets. For a test suite meant to qualify future
hardware, the eyes should look where color would pull them.

Design sketch (not yet built):
- Add a chroma-contrast channel to the salience map: center-surround on
  color-opponent axes (R/G, B/Y), the early-visual-system version.
- Weight it against the existing luminance + transient channels; keep the
  inhibition-of-return and POI machinery unchanged.
- The foveated-color render already desaturates the periphery, so the
  driver and the renderer would finally agree about what color is *for*:
  the driver uses it to decide where to look, the renderer uses it to
  decide what the look was worth.

Not blocking anything current. The grayscale driver is the honest baseline;
this is the upgrade that makes it human.
