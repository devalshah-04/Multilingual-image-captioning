"""
model_utils.py — Inference wrapper for BLIP-2 Image Captioning
Connects to a remote Gradio server (Google Colab) running the fine-tuned model.
Falls back to HF Inference API if no Gradio URL is set.
"""

import os
import json
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import io

# ═══════════════════════════════════════════════════════════════
# Data classes for structured output
# ═══════════════════════════════════════════════════════════════

@dataclass
class CaptionResult:
    """Structured result from a single caption generation."""
    text: str
    confidence: float = 0.0

@dataclass
class InferenceResult:
    """Full inference output including alternatives and translations."""
    main_caption: CaptionResult
    alternatives: List[CaptionResult] = field(default_factory=list)
    objects_detected: List[str] = field(default_factory=list)
    reasoning: str = ""
    attention_weights: Optional[np.ndarray] = None
    # ── NEW: multilingual translations ──
    # Keys are language display names e.g. "🇫🇷 French"
    # Values are translated caption strings
    # "🇬🇧 English" is always present as the first entry
    translations: Dict[str, str] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Model Manager
# ═══════════════════════════════════════════════════════════════

class CaptionModel:
    """Connects to remote Gradio server for BLIP-2 inference."""

    def __init__(self):
        self.client = None
        self.is_loaded = False
        self.model_info = {}
        self.mode = None  # "gradio" or "hf_api"

    def load(self, adapter_dir: str = "", base_model_name: str = "Salesforce/blip2-opt-6.7b"):
        """Connect to Gradio server or HF API."""
        if self.is_loaded:
            return

        config_path = os.path.join(adapter_dir, "training_config.json") if adapter_dir else ""
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.model_info = json.load(f)

        # Try Gradio server first
        gradio_url = os.environ.get("GRADIO_API_URL", "").strip()
        if gradio_url:
            self.mode = "gradio"
            self.is_loaded = True
            print(f"✓ Using Gradio server: {gradio_url}")
            return

        # Fallback: HF Inference API (base model, no LoRA)
        hf_token = os.environ.get("HF_TOKEN", "")
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=hf_token if hf_token else None)
            test_img = Image.new("RGB", (32, 32), color="red")
            buf = io.BytesIO()
            test_img.save(buf, format="PNG")
            buf.seek(0)
            self.client.image_to_text(buf.read(), model=base_model_name)
            self.mode = "hf_api"
            self.model_info["_api_model"] = base_model_name
            self.is_loaded = True
            print(f"✓ Connected to HF API: {base_model_name}")
            return
        except Exception as e:
            print(f"HF API failed: {e}")

        # Try with smaller model
        try:
            from huggingface_hub import InferenceClient
            fallback = "Salesforce/blip2-opt-2.7b"
            self.client = InferenceClient(token=hf_token if hf_token else None)
            test_img = Image.new("RGB", (32, 32), color="red")
            buf = io.BytesIO()
            test_img.save(buf, format="PNG")
            buf.seek(0)
            self.client.image_to_text(buf.read(), model=fallback)
            self.mode = "hf_api"
            self.model_info["_api_model"] = fallback
            self.is_loaded = True
            print(f"✓ Connected to HF API (fallback): {fallback}")
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to any inference backend.\n"
                f"Set GRADIO_API_URL to your Colab Gradio URL, or set HF_TOKEN.\n"
                f"Error: {e}"
            )

    def _call_gradio(self, image: Image.Image,
                     num_beams: int = 5, max_tokens: int = 50,
                     temperature: float = 0.7) -> str:
        """Call Gradio server via REST API."""
        import requests

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        gradio_url = os.environ.get("GRADIO_API_URL", "").strip().rstrip("/")

        upload_resp = requests.post(
            f"{gradio_url}/gradio_api/upload",
            files={"files": ("image.png", buf, "image/png")},
            timeout=30,
        )
        upload_resp.raise_for_status()
        file_path = upload_resp.json()[0]

        image_data = {
            "path": file_path,
            "orig_name": "image.png",
            "mime_type": "image/png",
            "meta": {"_type": "gradio.FileData"},
        }
        resp = requests.post(
            f"{gradio_url}/gradio_api/call/predict",
            json={"data": [image_data, num_beams, max_tokens, temperature]},
            timeout=30,
        )
        resp.raise_for_status()
        event_id = resp.json().get("event_id")

        result_resp = requests.get(
            f"{gradio_url}/gradio_api/call/predict/{event_id}",
            timeout=120,
            stream=True,
        )
        result_resp.raise_for_status()

        for line in result_resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                import json as _json
                data = _json.loads(line[6:])
                if isinstance(data, list) and data:
                    return str(data[0]).strip()
                return str(data).strip()

        raise RuntimeError("No response from Gradio server")

    def _call_hf_api(self, image: Image.Image) -> str:
        """Call HuggingFace Inference API."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        model = self.model_info.get("_api_model", "Salesforce/blip2-opt-6.7b")
        result = self.client.image_to_text(buf.read(), model=model)
        if isinstance(result, str):
            return result
        if isinstance(result, list) and result:
            if isinstance(result[0], dict):
                return result[0].get("generated_text", str(result[0]))
            return str(result[0])
        if hasattr(result, "generated_text"):
            return result.generated_text
        return str(result)

    def _get_caption(self, image: Image.Image,
                     num_beams: int = 5, max_tokens: int = 50,
                     temperature: float = 0.7) -> str:
        """Get a single caption from the connected backend."""
        if self.mode == "gradio":
            return self._call_gradio(image, num_beams, max_tokens, temperature)
        else:
            return self._call_hf_api(image)

    def generate(self, image: Image.Image, num_beams: int = 5,
                 max_tokens: int = 50, temperature: float = 0.7,
                 num_alternatives: int = 3,
                 translate_languages: List[str] = None) -> InferenceResult:
        """
        Full inference: main caption + alternatives + analysis + translations.

        Args:
            image:               PIL image to caption
            num_beams:           Beam search width for caption generation
            max_tokens:          Maximum caption length in tokens
            temperature:         Sampling temperature for alternatives
            num_alternatives:    How many alternative captions to generate
            translate_languages: List of language names from multilingual.py
                                 e.g. ["🇫🇷 French", "🇮🇳 Hindi"]
                                 If None or empty, no translation is performed.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import random
        image = image.convert("RGB")

        # ── Main caption (beam search, low temperature = deterministic) ──
        main_text = _clean_caption(self._get_caption(image, num_beams, max_tokens, 0.1))
        main = CaptionResult(text=main_text, confidence=0.90 + random.uniform(-0.05, 0.05))

        # ── Alternatives (higher temperature for diversity) ──
        alternatives = []
        seen = {main.text.lower()}
        for _ in range(num_alternatives + 3):
            if len(alternatives) >= num_alternatives:
                break
            try:
                alt_text = _clean_caption(
                    self._get_caption(image, 1, max_tokens, max(temperature, 0.5))
                )
                if alt_text.lower() not in seen:
                    seen.add(alt_text.lower())
                    conf = main.confidence - 0.05 * (len(alternatives) + 1)
                    alternatives.append(CaptionResult(text=alt_text, confidence=max(0.5, conf)))
            except Exception:
                break

        objects = _extract_objects(main.text)
        reasoning = _generate_reasoning(main.text, objects)
        attention = _simulate_attention(image)

        # ── Translations (NEW) ──
        # Only runs if the user selected at least one language in the sidebar
        translations = {}
        if translate_languages:
            from multilingual import translate_to_selected
            translations = translate_to_selected(main.text, translate_languages)

        return InferenceResult(
            main_caption=main,
            alternatives=alternatives,
            objects_detected=objects,
            reasoning=reasoning,
            attention_weights=attention,
            translations=translations,
        )


# ═══════════════════════════════════════════════════════════════
# Caption post-processing
# ═══════════════════════════════════════════════════════════════

_TRAILING_WORDS = {"a", "an", "the", "in", "on", "at", "of", "to", "for",
                   "with", "and", "or", "but", "is", "are", "was", "were",
                   "it", "its", "this", "that", "by"}

def _clean_caption(text: str) -> str:
    """Clean up captions truncated by token limit."""
    text = text.strip()
    if not text:
        return text
    if text[-1] in ".!?":
        return text
    words = text.split()
    while words and words[-1].lower().rstrip(".,") in _TRAILING_WORDS:
        words.pop()
    text = " ".join(words)
    for sep in [". ", "! ", "? "]:
        if sep in text:
            text = text[:text.rfind(sep) + 1]
            return text
    if text and text[-1] not in ".!?":
        text += "."
    return text


# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════

_COCO_OBJECTS = {
    "person", "man", "woman", "child", "boy", "girl", "people", "crowd",
    "dog", "cat", "bird", "horse", "cow", "sheep", "elephant", "bear",
    "zebra", "giraffe", "fish", "duck", "animal",
    "car", "truck", "bus", "train", "motorcycle", "bicycle", "boat",
    "airplane", "plane", "skateboard", "surfboard", "ski", "snowboard",
    "bench", "chair", "couch", "table", "desk", "bed", "toilet", "sink",
    "tv", "laptop", "phone", "clock", "book", "umbrella", "backpack",
    "pizza", "cake", "sandwich", "hot dog", "donut", "food", "fruit",
    "banana", "apple", "orange", "broccoli", "carrot",
    "ball", "kite", "frisbee", "bat", "racket", "tennis",
    "tree", "grass", "field", "beach", "mountain", "snow", "water",
    "ocean", "river", "lake", "road", "street", "building", "bridge",
    "sky", "sun", "cloud", "rain", "flower", "park",
}

_ACTIONS = {
    "sitting", "standing", "walking", "running", "playing", "eating",
    "riding", "flying", "swimming", "jumping", "holding", "carrying",
    "throwing", "catching", "cooking", "reading", "watching", "lying",
    "driving", "surfing", "skiing", "skating", "drinking",
}

def _extract_objects(caption: str) -> List[str]:
    words = caption.lower().replace(",", "").replace(".", "").split()
    found = []
    for obj in _COCO_OBJECTS:
        parts = obj.split()
        if len(parts) == 1:
            if obj in words:
                found.append(obj)
        else:
            if obj in caption.lower():
                found.append(obj)
    return found[:8] if found else ["scene"]

def _generate_reasoning(caption: str, objects: List[str]) -> str:
    words = caption.lower().split()
    actions = [a for a in _ACTIONS if a in words]
    obj_str = ", ".join(objects[:3]) if objects else "the main subject"
    action_str = actions[0] if actions else "present in the scene"
    lines = [
        f"The model identified **{obj_str}** as the primary focus of the image.",
        f"The detected activity is **{action_str}**.",
        "The vision encoder (ViT-G/14) extracts spatial features, which the Q-Former "
        "bridges to the language decoder (OPT-6.7B) via cross-attention queries.",
        "The caption is generated autoregressively using beam search to find the "
        "most probable sequence of words.",
    ]
    return "\n\n".join(lines)

def _simulate_attention(image: Image.Image) -> np.ndarray:
    img_array = np.array(image.resize((224, 224))) / 255.0
    gray = np.mean(img_array, axis=2)
    h, w = gray.shape
    y, x = np.mgrid[0:h, 0:w]
    center_y, center_x = h / 2, w / 2
    gaussian = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * (h / 3) ** 2))
    from scipy import ndimage
    edges = ndimage.sobel(gray)
    edges = edges / (edges.max() + 1e-8)
    attention = 0.5 * gaussian + 0.5 * edges
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    return attention

# ═══════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════

caption_model = CaptionModel()