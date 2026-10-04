"""Human temporal-vision baseline: the timing budget the pipeline emulates.

Consensus values from vision science. These are the demonstrative baseline,
not fitted parameters -- change them and the pipeline should stop behaving
like a human (which is itself a test).
"""

# Photon -> retinal ganglion signal
PHOTOTRANSDUCTION_MS = 35.0
# Retina -> usable cortical signal (LGN -> V1)
RETINA_TO_CORTEX_MS = 65.0
# Total pipeline latency: you live this far in the past
PIPELINE_LATENCY_MS = PHOTOTRANSDUCTION_MS + RETINA_TO_CORTEX_MS  # 100

# Full recognition / categorization latency
RECOGNITION_MS = 150.0

# Bloch's law: luminance integrates over ~100 ms
INTEGRATION_WINDOW_MS = 100.0

# Alpha-cycle discretization (modeling assumption, provisional):
# perceptual moments discretized at ~10 Hz, after the alpha cycle
PERCEPTUAL_MOMENT_MS = 100.0

# Saccades: 3-4 ballistic jumps per second
SACCADE_RATE_HZ = 3.5

# Visually-guided saccade latency: time from "decide" to eyes moving
SACCADE_LATENCY_MS = 200.0

# Smooth pursuit: the eye-glide that tracks a moving target. No
# saccadic suppression during pursuit -- the eyes stay online while
# tracking, unlike the ballistic jump.
PURSUIT_LATENCY_MS = 100.0   # faster than a saccade: pursuit is cheap
PURSUIT_SEGMENT_MS = 1000.0  # pursuit is planned in 1 s segments,
                             # re-evaluated at each decision tick
PURSUIT_GAIN = 1.0           # pursuit velocity / target velocity

# Fovea: ~2 degrees of high acuity; e2 = eccentricity falloff constant
FOVEA_RADIUS_DEG = 1.0
E2_DEG = 2.5

# Foveal flicker fusion threshold
FLICKER_FUSION_HZ = 60.0

# Modeled visual field width (degrees) mapped onto the frame
FIELD_WIDTH_DEG = 40.0
