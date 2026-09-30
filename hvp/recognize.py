"""Zero-shot object recognition for foveal crops.

FovealClassifier: CLIP (transformers, ViT-B/32) zero-shot over a
caller-supplied vocabulary. The model loads lazily on first use;
weights live in the HuggingFace cache (not in git).

The vocabulary is the caller's business: pass scene-specific labels
("gate", "dark", "sign") for the top-down loop, or GENERAL_VOCAB for
open-world naming. Below `unknown_threshold` the classifier honestly
returns "unknown" instead of guessing.
"""

import os

MODEL_ID = os.environ.get("HVP_CLIP_MODEL", "openai/clip-vit-base-patch32")

GENERAL_VOCAB = [
    "person", "face", "car", "dog", "cat", "tree", "building",
    "window", "door", "sign", "street light", "table", "chair",
    "television", "phone", "book", "cup", "bottle", "road",
    "sky", "crowd",
]

# Scene-specific vocabulary for the dark street / gate scene (Star Tours
# ride film). Prompts tuned 2026-09-30 against 12 hand-labeled foveal
# crops: 6/12 top-1 agreement (scripts/audit_recognition.py); remaining
# misses are dark/ambiguous crops where the human labeler had temporal
# context the single crop lacks. Grayscale phrasing matches the
# pipeline's crop domain.
DARK_STREET_VOCAB = {
    "gate": "a grayscale photo of a metal security gate with horizontal slats",
    "gate-edge": "a grayscale photo of the edge of a metal gate",
    "dark": "a grayscale photo of complete darkness, nothing visible",
    "light-strip": "a grayscale photo of a single vertical strip of light",
    "distant-light": "a grayscale photo of one small distant glowing light",
    "sign": "a grayscale photo of an illuminated sign with text",
    "lit-floor": "a grayscale photo of a floor lit by overhead lights",
    "windows": "a grayscale photo of a row of lit windows",
}


def prompt_for(name: str) -> str:
    return f"a photo of {name}"


class FovealClassifier:
    """Zero-shot CLIP classifier over foveal crops.

    classify(crop, labels) -> (name, confidence). crop is a PIL Image.
    labels is a list of names, or (name, prompt) pairs for phrasing
    control. Confidence is the softmax probability of the winner;
    winners below unknown_threshold come back as "unknown".
    """

    def __init__(self, model_id: str = MODEL_ID,
                 unknown_threshold: float = 0.30):
        self.model_id = model_id
        self.unknown_threshold = unknown_threshold
        self._model = None
        self._processor = None
        self._text_cache: dict[tuple, object] = {}

    def _ensure(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor
            self._processor = CLIPProcessor.from_pretrained(self.model_id)
            self._model = CLIPModel.from_pretrained(self.model_id)
            self._model.eval()

    def _text_features(self, labels):
        import torch
        key = tuple(p for _, p in labels)
        if key not in self._text_cache:
            inputs = self._processor(
                text=[p for _, p in labels],
                return_tensors="pt", padding=True)
            with torch.no_grad():
                feats = self._model.get_text_features(**inputs).pooler_output
            self._text_cache[key] = feats / feats.norm(dim=-1, keepdim=True)
        return self._text_cache[key]

    def _normalize(self, labels) -> list[tuple[str, str]]:
        """labels -> [(name, prompt)]; bare names get prompt_for()."""
        pairs = []
        for l in labels:
            if isinstance(l, tuple):
                name, prompt = l
                pairs.append((name, prompt or prompt_for(name)))
            else:
                pairs.append((l, prompt_for(l)))
        return pairs

    def classify(self, crop, labels) -> tuple[str, float]:
        """Classify one foveal crop. Returns (label, confidence)."""
        import torch
        self._ensure()
        pairs = self._normalize(labels)
        inputs = self._processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            img = self._model.get_image_features(**inputs).pooler_output
            img = img / img.norm(dim=-1, keepdim=True)
            text = self._text_features(pairs)
            logits = (img @ text.T) * self._model.logit_scale.exp()
            probs = logits.softmax(dim=-1)[0]
        best = int(probs.argmax())
        conf = float(probs[best])
        name = pairs[best][0]
        if conf < self.unknown_threshold:
            return "unknown", conf
        return name, conf

    def distribution(self, crop, labels) -> list[tuple[str, float]]:
        """Full (label, probability) ranking, for audits."""
        import torch
        self._ensure()
        pairs = self._normalize(labels)
        inputs = self._processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            img = self._model.get_image_features(**inputs).pooler_output
            img = img / img.norm(dim=-1, keepdim=True)
            text = self._text_features(pairs)
            logits = (img @ text.T) * self._model.logit_scale.exp()
            probs = logits.softmax(dim=-1)[0]
        ranked = sorted(zip([n for n, _ in pairs], probs.tolist()),
                        key=lambda t: -t[1])
        return ranked
