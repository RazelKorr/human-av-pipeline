"""Streaming harness for the Sensorium v2 color-saccades pipeline.

Wraps the batch v2 driver (scripts/run_video_saccades_color.py) in a
live-like streaming loop WITHOUT rewriting the pipeline itself:

  feeder         -- test video played as if live: ONE ffmpeg decode by
                    default (split filter graph yields the 224px
                    attention frame + the work-res frame + optionally
                    the motion thumbnail; the 224px frames are
                    bit-identical to a dual decode). --dual-decode
                    keeps the original two-lockstep-pipes validation
                    configuration reproducible.
  online_driver  -- OnlineAttentionDriver: the batch pass-1 attention
                    math as a stateful object (saccade decisions online,
                    200 ms latency)
  online_render  -- OnlineVisionPipeline: VisionPipeline subclass with a
                    bounded frame buffer; moments pulled as stream time
                    advances

Pre-hardware, test-fixture only. No live sensors.
"""
