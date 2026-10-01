"""On-demand open-vocabulary detection (OWL-ViT).

The foveal classifier only names what gaze has already fixated. When
language asks for something never seen ('where is the X', or 'look at
the X' with no track in memory), the detector scans the current frame
once and proposes boxes.

Slow (~1-2 s per frame on CPU): on-demand only, never in the 10 Hz
loop. Same zero-shot philosophy as the CLIP classifier -- the query
is free text, so there is no fixed label set.
"""
import os

_MODEL_ID = "google/owlvit-base-patch32"


def pil_from_frame(frame):
    """224x224 float32 grayscale frame (0..1) -> PIL RGB.

    The runner's frames are grayscale float32; the detector takes PIL
    RGB. Single conversion point so the runner and the tests agree on
    exactly what bytes a live query sees.
    """
    import numpy as np
    from PIL import Image
    arr = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


class ObjectDetector:
    def __init__(self, model_id: str = _MODEL_ID):
        self.model_id = model_id
        self._processor = None
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        # Bracketed IPv6 literals in no_proxy (e.g. [::1]) break
        # httpx's proxy parsing; HF hub goes through httpx. Reset both
        # vars rather than dumping the environment (proxy URLs may
        # carry credentials).
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        os.environ["no_proxy"] = "localhost,127.0.0.1"
        from transformers import (OwlViTProcessor,
                                  OwlViTForObjectDetection)
        self._processor = OwlViTProcessor.from_pretrained(self.model_id)
        self._model = OwlViTForObjectDetection.from_pretrained(
            self.model_id)
        self._model.eval()

    def detect(self, frame_rgb, queries: list[str],
               threshold: float = 0.10):
        """frame_rgb: PIL image (224x224). queries: free-text phrases.

        Returns [(label, x0, y0, x1, y1, score)] in 224px coords,
        best first. Empty list when nothing passes the threshold.
        """
        self._ensure()
        import torch
        inputs = self._processor(text=[[q] for q in queries],
                                 images=frame_rgb, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        w, h = frame_rgb.size
        results = self._processor.post_process_grounded_object_detection(
            outputs, threshold=threshold, target_sizes=[(h, w)],
            text_labels=[list(queries)])
        dets = []
        for res in results:
            for box, score, li in zip(res["boxes"], res["scores"],
                                      res["labels"]):
                x0, y0, x1, y1 = (float(v) for v in box.tolist())
                dets.append((queries[int(li)], x0, y0, x1, y1,
                             float(score)))
        dets.sort(key=lambda d: d[5], reverse=True)
        return dets

    @staticmethod
    def box_center_map(det) -> tuple[float, float]:
        """Box center in 56x56 map coords."""
        _, x0, y0, x1, y1, _ = det
        return ((x0 + x1) / 2.0 / 4.0, (y0 + y1) / 2.0 / 4.0)
