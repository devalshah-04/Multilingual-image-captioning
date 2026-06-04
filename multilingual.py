"""
multilingual.py — Neural Machine Translation for multilingual image captions.

Uses Helsinki-NLP MarianMT models — pretrained seq2seq Transformers trained
on the OPUS corpus (millions of parallel sentence pairs per language pair).

Architecture per language pair:
    English caption
        → Tokenizer (splits into subword tokens)
        → MarianMT Encoder (self-attention over all tokens)
        → MarianMT Decoder (cross-attention + autoregressive generation)
        → Target language caption

Models are downloaded from HuggingFace Hub on first use (~300MB each)
and cached locally at ~/.cache/huggingface/
GPU is used automatically if available, otherwise CPU.
"""

import torch
from transformers import MarianMTModel, MarianTokenizer
from typing import Dict, List
import threading

# ═══════════════════════════════════════════════════════════════
# Language Registry
# Key   = display name shown in the UI
# Value = HuggingFace model ID for that language pair
# ═══════════════════════════════════════════════════════════════

SUPPORTED_LANGUAGES: Dict[str, str] = {
    "🇫🇷 French":     "Helsinki-NLP/opus-mt-en-fr",
    "🇪🇸 Spanish":    "Helsinki-NLP/opus-mt-en-es",
    "🇩🇪 German":     "Helsinki-NLP/opus-mt-en-de",
    "🇮🇳 Hindi":      "Helsinki-NLP/opus-mt-en-hi",
    "🇯🇵 Japanese":   "Helsinki-NLP/opus-mt-en-jap",
    "🇨🇳 Chinese":    "Helsinki-NLP/opus-mt-en-zh",
    "🇵🇹 Portuguese": "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "🇷🇺 Russian":    "Helsinki-NLP/opus-mt-en-ru",
    "🇮🇹 Italian":    "Helsinki-NLP/opus-mt-en-it",
    "🇦🇷 Arabic":     "Helsinki-NLP/opus-mt-en-ar",
}

# Native script labels shown in the UI alongside translations
LANGUAGE_NATIVE: Dict[str, str] = {
    "🇫🇷 French":     "Français",
    "🇪🇸 Spanish":    "Español",
    "🇩🇪 German":     "Deutsch",
    "🇮🇳 Hindi":      "हिन्दी",
    "🇯🇵 Japanese":   "日本語",
    "🇨🇳 Chinese":    "中文",
    "🇵🇹 Portuguese": "Português",
    "🇷🇺 Russian":    "Русский",
    "🇮🇹 Italian":    "Italiano",
    "🇦🇷 Arabic":     "العربية",
}

# ═══════════════════════════════════════════════════════════════
# Model Cache
# Models loaded once per session, reused across all calls.
# First load per language: ~5-10 seconds (download + init)
# Subsequent calls: instant (served from cache)
# ═══════════════════════════════════════════════════════════════

_model_cache: Dict[str, tuple] = {}   # lang → (tokenizer, model)
_cache_lock = threading.Lock()
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(language: str) -> tuple:
    """
    Load and cache the MarianMT model for a given language.
    Thread-safe via lock to prevent double-loading.
    """
    with _cache_lock:
        if language not in _model_cache:
            model_id = SUPPORTED_LANGUAGES[language]
            tokenizer = MarianTokenizer.from_pretrained(model_id)
            model = MarianMTModel.from_pretrained(model_id)
            model = model.to(_device)
            model.eval()
            _model_cache[language] = (tokenizer, model)
        return _model_cache[language]


# ═══════════════════════════════════════════════════════════════
# Core Translation Function
# ═══════════════════════════════════════════════════════════════

def translate(text: str, language: str) -> str:
    """
    Translate an English caption to the target language.

    Pipeline:
    1. Tokenizer splits English text into subword tokens
    2. Encoder reads all tokens via self-attention → context vectors
    3. Decoder generates target tokens autoregressively via cross-attention
    4. Beam search (4 beams) picks the best candidate sequence
    5. Tokenizer decodes output IDs → readable string

    Args:
        text:     English caption string from BLIP-2
        language: Key from SUPPORTED_LANGUAGES (e.g. "🇫🇷 French")

    Returns:
        Translated string, or error message if translation fails
    """
    if not text or not text.strip():
        return ""

    if language not in SUPPORTED_LANGUAGES:
        return f"[Unsupported language: {language}]"

    try:
        tokenizer, model = _load_model(language)

        inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            translated_ids = model.generate(
                **inputs,
                num_beams=4,
                max_length=128,
                early_stopping=True,
            )

        return tokenizer.decode(translated_ids[0], skip_special_tokens=True).strip()

    except Exception as e:
        return f"[Translation error: {str(e)[:80]}]"


def translate_to_selected(text: str, languages: List[str]) -> Dict[str, str]:
    """
    Translate caption into multiple selected languages.

    Returns ordered dict — English always first, then each selected language.
    Example:
        {
            "🇬🇧 English": "A squirrel eating a nut on a rock.",
            "🇫🇷 French":  "Un écureuil mangeant une noix sur un rocher.",
            "🇮🇳 Hindi":   "एक गिलहरी एक चट्टान पर अखरोट खा रही है।"
        }
    """
    results = {"🇬🇧 English": text}
    for lang in languages:
        results[lang] = translate(text, lang)
    return results


def get_language_names() -> List[str]:
    """Return all supported language display names."""
    return list(SUPPORTED_LANGUAGES.keys())