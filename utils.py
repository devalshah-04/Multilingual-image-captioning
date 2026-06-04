"""
utils.py — UI helpers for the Streamlit Image Captioning app.
Handles image loading, attention overlay rendering, theme CSS, and history.
"""

import io
import json
import os
import time
import base64
import hashlib
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import Optional, List, Dict
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# Image Loading
# ═══════════════════════════════════════════════════════════════

def load_image_from_url(url: str, timeout: int = 10) -> Optional[Image.Image]:
    """Download and open an image from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not load image from URL: {e}")


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL image to base64 data URI for HTML embedding."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def image_hash(image: Image.Image) -> str:
    """Quick perceptual hash for deduplication."""
    thumb = image.resize((8, 8)).convert("L")
    return hashlib.md5(thumb.tobytes()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════
# Attention Heatmap Overlay
# ═══════════════════════════════════════════════════════════════

def render_attention_overlay(image: Image.Image, attention: np.ndarray,
                              alpha: float = 0.45) -> Image.Image:
    """Overlay an attention heatmap on the original image."""
    img_resized = image.resize((attention.shape[1], attention.shape[0]))
    img_array = np.array(img_resized) / 255.0
    colormap = cm.get_cmap("magma")
    heatmap_rgba = colormap(attention)[:, :, :3]
    blended = (1 - alpha) * img_array + alpha * heatmap_rgba
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ═══════════════════════════════════════════════════════════════
# History Manager
# ═══════════════════════════════════════════════════════════════

HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caption_history")
HISTORY_INDEX = os.path.join(HISTORY_DIR, "index.json")

# ── Auto-classification keywords ──
_CATEGORIES = {
    "Insect": ["butterfly", "bee", "ant", "spider", "dragonfly", "moth",
               "beetle", "ladybug", "caterpillar", "fly", "mosquito",
               "grasshopper", "cricket", "wasp", "insect", "bug"],
    "Aquatic": ["fish", "whale", "dolphin", "shark", "octopus", "jellyfish",
                "coral", "reef", "underwater", "aquarium", "sea", "swim",
                "seal", "starfish", "crab", "lobster", "shrimp", "pond"],
    "Aerial": ["airplane", "helicopter", "drone", "kite", "parachute",
               "balloon", "jet", "glider", "flying", "airshow", "pilot"],
    "Animal": ["dog", "cat", "bird", "horse", "elephant", "bear",
               "giraffe", "zebra", "cow", "sheep", "lion", "tiger", "animal",
               "monkey", "rabbit", "duck", "chicken", "frog", "turtle",
               "deer", "squirrel", "rabbit", "penguin", "parrot", "puppy",
               "kitten", "panda", "fox", "wolf", "hamster", "mouse", "goat", "pig"],
    "Food": ["food", "plate", "pizza", "cake", "sandwich", "fruit", "banana",
             "apple", "bowl", "salad", "meal", "dish", "restaurant", "eating",
             "kitchen", "dining", "coffee", "drink", "wine", "bread",
             "donut", "burger", "sushi", "ice cream", "chocolate", "soup",
             "cookie", "noodle", "rice", "cheese", "meat", "vegetable"],
    "Nature": ["tree", "mountain", "river", "ocean", "lake", "forest", "sky",
               "sunset", "sunrise", "beach", "field", "flower", "garden",
               "landscape", "grass", "cloud", "snow", "rain", "waterfall",
               "hill", "valley", "cliff", "island", "desert", "jungle",
               "meadow", "swamp", "volcano", "canyon", "creek", "pond"],
    "Weather": ["storm", "lightning", "thunder", "rainbow", "fog", "mist",
                "tornado", "hurricane", "blizzard", "hail", "frost",
                "sunny", "cloudy", "rainy", "snowy", "windy"],
    "Vehicle": ["car", "bus", "truck", "train", "boat", "ship",
                "motorcycle", "bicycle", "bike", "vehicle", "taxi", "van",
                "ambulance", "firetruck", "tractor", "scooter", "subway",
                "ferry", "yacht", "canoe", "kayak", "sailboat"],
    "Sports": ["ball", "tennis", "baseball", "soccer", "football", "surfboard",
               "skateboard", "ski", "snowboard", "frisbee", "bat", "racket",
               "court", "stadium", "playing", "game", "team", "basketball",
               "volleyball", "hockey", "golf", "boxing", "wrestling",
               "swimming", "running", "cycling", "gymnast", "athlete"],
    "Technology": ["laptop", "computer", "phone", "robot", "screen", "monitor",
                   "keyboard", "camera", "drone", "tablet", "headphone",
                   "gadget", "device", "server", "circuit", "wire"],
    "Indoor": ["room", "table", "chair", "desk", "bed", "couch", "sofa",
               "television", "tv", "clock", "bathroom", "bedroom", "office",
               "shelf", "book", "lamp", "curtain", "carpet", "mirror",
               "wardrobe", "fireplace", "staircase", "hallway"],
    "Architecture": ["church", "temple", "mosque", "castle", "palace",
                     "monument", "statue", "fountain", "dome", "arch",
                     "cathedral", "pyramid", "ruins", "tower", "lighthouse"],
    "Urban": ["building", "city", "bridge", "sign", "store", "shop",
              "market", "station", "airport", "sidewalk", "window",
              "road", "street", "traffic", "parking", "highway",
              "skyscraper", "billboard", "crosswalk", "alley", "plaza"],
    "Art": ["painting", "sculpture", "mural", "graffiti", "drawing",
            "sketch", "canvas", "gallery", "museum", "artwork", "mosaic",
            "portrait", "abstract", "illustration", "poster"],
    "Human": ["man", "woman", "person", "people", "boy", "girl", "child", "baby",
              "crowd", "player", "wearing", "couple", "group", "selfie"],
}

_WEAK_HUMAN = {"sitting", "standing", "walking", "smiling", "holding"}


def classify_caption(caption: str) -> str:
    """Classify a caption into a category based on keywords."""
    words = set(caption.lower().replace(".", "").replace(",", "").split())
    scores = {}
    for cat, keywords in _CATEGORIES.items():
        scores[cat] = sum(1 for k in keywords if k in words)
    non_human_max = max((v for k, v in scores.items() if k != "Human"), default=0)
    if non_human_max == 0:
        scores["Human"] += sum(1 for w in _WEAK_HUMAN if w in words)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


class HistoryManager:
    """Manages caption history in session state + persistent disk storage."""

    @staticmethod
    def _ensure_dir():
        os.makedirs(HISTORY_DIR, exist_ok=True)

    @staticmethod
    def _load_index() -> List[Dict]:
        if os.path.exists(HISTORY_INDEX):
            with open(HISTORY_INDEX, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    @staticmethod
    def _save_index(entries: List[Dict]):
        with open(HISTORY_INDEX, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    @staticmethod
    def add(session_state, image: Image.Image, caption: str,
            confidence: float, alternatives: List[str],
            translations: Dict[str, str] = None):
        """
        Add a new entry to history (skips duplicates). Saves to disk.
        translations: dict of {language_name: translated_caption}
                      e.g. {"🇬🇧 English": "...", "🇫🇷 French": "..."}
        """
        if "history" not in session_state:
            session_state["history"] = []

        # Skip if same caption as most recent entry
        if session_state["history"]:
            last = session_state["history"][0]
            if last["caption"].strip().lower() == caption.strip().lower():
                last["confidence"] = confidence
                last["alternatives"] = alternatives
                last["timestamp"] = datetime.now().strftime("%H:%M:%S")
                # Update translations if provided
                if translations:
                    last["translations"] = translations
                return

        HistoryManager._ensure_dir()

        category = classify_caption(caption)

        # Save image to category folder
        cat_dir = os.path.join(HISTORY_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"{ts}.jpg"
        img_path = os.path.join(cat_dir, img_filename)
        image.save(img_path, "JPEG", quality=90)

        # Create thumbnail
        thumb = image.copy()
        thumb.thumbnail((80, 80))

        # Session state entry
        entry = {
            "id": len(session_state["history"]),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "thumbnail": image_to_base64(thumb, "JPEG"),
            "image": image,
            "caption": caption,
            "confidence": confidence,
            "alternatives": alternatives,
            "category": category,
            "image_path": img_path,
            # ── NEW: store translations in session history ──
            "translations": translations or {},
        }
        session_state["history"].insert(0, entry)

        if len(session_state["history"]) > 50:
            session_state["history"] = session_state["history"][:50]

        # Save to disk index — includes translations
        disk_entries = HistoryManager._load_index()
        disk_entry = {
            "timestamp": entry["timestamp"],
            "date": entry["date"],
            "caption": caption,
            "confidence": confidence,
            "alternatives": alternatives,
            "category": category,
            "image_path": img_path,
            # ── NEW: persist translations to index.json ──
            "translations": translations or {},
        }
        disk_entries.insert(0, disk_entry)
        if len(disk_entries) > 100:
            disk_entries = disk_entries[:100]
        HistoryManager._save_index(disk_entries)

    @staticmethod
    def load_from_disk(session_state):
        """Load history from disk into session state (on app startup)."""
        if session_state.get("_disk_loaded"):
            return
        session_state["_disk_loaded"] = True

        disk_entries = HistoryManager._load_index()
        if not disk_entries:
            return

        history = session_state.get("history", [])
        for de in disk_entries:
            img_path = de.get("image_path", "")
            if not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                thumb = img.copy()
                thumb.thumbnail((80, 80))
                entry = {
                    "id": len(history),
                    "timestamp": de.get("timestamp", ""),
                    "date": de.get("date", ""),
                    "thumbnail": image_to_base64(thumb, "JPEG"),
                    "image": img,
                    "caption": de["caption"],
                    "confidence": de["confidence"],
                    "alternatives": de.get("alternatives", []),
                    "category": de.get("category", "Other"),
                    "image_path": img_path,
                    # ── NEW: load translations from disk ──
                    "translations": de.get("translations", {}),
                }
                history.append(entry)
            except Exception:
                continue

        session_state["history"] = history

    @staticmethod
    def get_all(session_state) -> List[Dict]:
        return session_state.get("history", [])

    @staticmethod
    def clear(session_state):
        session_state["history"] = []


# ═══════════════════════════════════════════════════════════════
# Confidence Gauge Renderer
# ═══════════════════════════════════════════════════════════════

def confidence_color(score: float) -> str:
    if score >= 0.85:
        return "#22c55e"
    elif score >= 0.70:
        return "#eab308"
    else:
        return "#ef4444"


def confidence_label(score: float) -> str:
    if score >= 0.85:
        return "High"
    elif score >= 0.70:
        return "Medium"
    else:
        return "Low"


# ═══════════════════════════════════════════════════════════════
# Typing Animation HTML
# ═══════════════════════════════════════════════════════════════

def typing_animation_html(text: str, speed_ms: int = 30) -> str:
    escaped = text.replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n")
    uid = f"caption_{hash(text) % 100000}"
    return f"""
    <div id="{uid}" style="font-size:1.35rem;font-weight:500;line-height:1.7;color:inherit;min-height:2.5rem;"></div>
    <script>
    (function() {{
        const el = document.getElementById('{uid}');
        const text = '{escaped}';
        let i = 0;
        el.textContent = '';
        const interval = setInterval(() => {{
            if (i < text.length) {{ el.textContent += text[i]; i++; }}
            else {{ clearInterval(interval); }}
        }}, {speed_ms});
    }})();
    </script>
    """


# ═══════════════════════════════════════════════════════════════
# CSS Themes
# ═══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

.ui-card {
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(128, 128, 128, 0.15);
    backdrop-filter: blur(10px);
    transition: box-shadow 0.3s ease, transform 0.2s ease;
}
.ui-card:hover { box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12); }

.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-purple { background: rgba(139, 92, 246, 0.12); color: #8b5cf6; }
.badge-blue   { background: rgba(59, 130, 246, 0.12); color: #3b82f6; }
.badge-green  { background: rgba(34, 197, 94, 0.12);  color: #22c55e; }
.badge-amber  { background: rgba(245, 158, 11, 0.12); color: #f59e0b; }

.gradient-text {
    background: linear-gradient(135deg, #8b5cf6, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.step-item { display: flex; align-items: center; gap: 10px; padding: 8px 0; font-size: 0.9rem; }
.step-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.step-active  .step-dot { background: #8b5cf6; box-shadow: 0 0 8px rgba(139,92,246,0.5); }
.step-done    .step-dot { background: #22c55e; }
.step-pending .step-dot { background: #6b7280; opacity: 0.4; }

.conf-bar-bg { width: 100%; height: 6px; border-radius: 3px; background: rgba(128,128,128,0.15); overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 3px; transition: width 0.8s ease; }

.history-card {
    display: flex; gap: 12px; align-items: center;
    padding: 10px 12px; border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.12);
    cursor: pointer; transition: background 0.2s; margin-bottom: 8px;
}
.history-card:hover { background: rgba(139, 92, 246, 0.06); }
.history-card img { width: 48px; height: 48px; border-radius: 8px; object-fit: cover; }
.history-card .caption-preview {
    font-size: 0.82rem; line-height: 1.35; overflow: hidden;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.history-card .time-label { font-size: 0.7rem; opacity: 0.45; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%) !important;
    border: none !important; color: white !important; font-weight: 600 !important;
    border-radius: 12px !important; padding: 0.6rem 2rem !important;
    box-shadow: 0 4px 14px rgba(139, 92, 246, 0.35) !important;
    transition: all 0.25s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(139, 92, 246, 0.5) !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(128,128,128,0.12); }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 20px; font-weight: 500; }

.alt-caption {
    padding: 12px 16px; border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.12);
    margin-bottom: 8px; font-size: 0.92rem; line-height: 1.5;
    display: flex; justify-content: space-between; align-items: center;
}
.alt-caption .conf-badge {
    font-size: 0.72rem; font-weight: 600;
    padding: 2px 10px; border-radius: 100px; white-space: nowrap;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in { animation: fadeInUp 0.4s ease-out forwards; }

.obj-tag {
    display: inline-block; padding: 4px 12px; border-radius: 8px;
    font-size: 0.8rem; font-weight: 500; margin: 3px 4px 3px 0;
    border: 1px solid rgba(128,128,128,0.15);
}

/* ── Translation cards (NEW) ── */
.translation-card {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 14px 18px;
    border-radius: 14px;
    border: 1px solid rgba(128, 128, 128, 0.13);
    margin-bottom: 10px;
    transition: background 0.2s ease;
    animation: fadeInUp 0.35s ease-out forwards;
}
.translation-card:hover {
    background: rgba(139, 92, 246, 0.04);
    border-color: rgba(139, 92, 246, 0.25);
}
.translation-card .lang-flag {
    font-size: 1.6rem;
    line-height: 1;
    flex-shrink: 0;
    margin-top: 2px;
}
.translation-card .lang-body {
    flex: 1;
}
.translation-card .lang-name {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    opacity: 0.5;
    text-transform: uppercase;
    margin-bottom: 3px;
}
.translation-card .lang-text {
    font-size: 1.0rem;
    font-weight: 500;
    line-height: 1.6;
}
.translation-card.english {
    border-left: 4px solid #8b5cf6;
    background: rgba(139, 92, 246, 0.04);
}
.translation-no-selection {
    text-align: center;
    padding: 50px 20px;
    opacity: 0.4;
    font-size: 0.95rem;
}
</style>
"""

# ═══════════════════════════════════════════════════════════════
# Loading Step Renderer
# ═══════════════════════════════════════════════════════════════

LOADING_STEPS = [
    ("🔍", "Extracting visual features..."),
    ("🧠", "Generating caption..."),
    ("✨", "Finalizing output..."),
]


def render_loading_step(current: int) -> str:
    html = '<div style="padding: 8px 0;">'
    for i, (icon, label) in enumerate(LOADING_STEPS):
        if i < current:
            cls, check = "step-done", "✓"
        elif i == current:
            cls, check = "step-active", icon
        else:
            cls, check = "step-pending", "○"
        html += f'<div class="step-item {cls}"><span class="step-dot"></span>{check} {label}</div>'
    html += "</div>"
    return html


# ═══════════════════════════════════════════════════════════════
# Translation Cards Renderer (NEW)
# ═══════════════════════════════════════════════════════════════

def render_translation_cards(translations: Dict[str, str]) -> str:
    """
    Render translation results as styled HTML cards.
    Each card shows: flag emoji + language name + translated caption.
    English is always shown first with a purple left border accent.

    Args:
        translations: Dict from translate_to_selected()
                      e.g. {"🇬🇧 English": "...", "🇫🇷 French": "..."}
    Returns:
        HTML string ready for st.markdown(..., unsafe_allow_html=True)
    """
    if not translations:
        return ""

    html = ""
    for i, (lang, text) in enumerate(translations.items()):
        # Split "🇬🇧 English" → flag="🇬🇧", name="English"
        parts = lang.split(" ", 1)
        flag = parts[0] if len(parts) > 0 else "🌐"
        name = parts[1] if len(parts) > 1 else lang

        is_english = "English" in lang
        card_class = "translation-card english" if is_english else "translation-card"
        delay = f"animation-delay: {i * 0.06}s;"

        html += f"""
        <div class="{card_class}" style="{delay}">
            <div class="lang-flag">{flag}</div>
            <div class="lang-body">
                <div class="lang-name">{name}</div>
                <div class="lang-text">{text}</div>
            </div>
        </div>
        """

    return html


# ═══════════════════════════════════════════════════════════════
# Sample Demo Data
# ═══════════════════════════════════════════════════════════════

DEMO_SAMPLES = [
    {
        "url": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400",
        "ground_truth": "A brown dog playing with a ball in a grassy park.",
        "fallback_pred": "a dog is playing in the grass with a ball",
    },
    {
        "url": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400",
        "ground_truth": "A busy city street with tall buildings and cars.",
        "fallback_pred": "a busy city street with tall buildings and traffic",
    },
    {
        "url": "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=400",
        "ground_truth": "A plate of delicious food with vegetables and sauce.",
        "fallback_pred": "a plate of food with vegetables on a table",
    },
]