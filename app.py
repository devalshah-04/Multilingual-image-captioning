"""
🖼️ Image Captioning AI — Streamlit App
Premium UI powered by BLIP-2 fine-tuned on MS-COCO.

Run:  streamlit run app.py
"""

import os
import sys
import time
import streamlit as st
from PIL import Image

# Local imports
from model_utils import caption_model, InferenceResult
from utils import (
    CUSTOM_CSS, load_image_from_url, image_to_base64,
    render_attention_overlay, HistoryManager,
    confidence_color, confidence_label, typing_animation_html,
    render_loading_step, render_translation_cards, DEMO_SAMPLES, LOADING_STEPS,
)
from multilingual import get_language_names, LANGUAGE_NATIVE

# ═══════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Image Captioning AI",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# Session State Init
# ═══════════════════════════════════════════════════════════════

if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "_uploader_key" not in st.session_state:
    st.session_state._uploader_key = 0
# ── NEW: default selected languages ──
if "selected_languages" not in st.session_state:
    st.session_state.selected_languages = ["🇫🇷 French", "🇮🇳 Hindi", "🇪🇸 Spanish"]


# ═══════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════

ADAPTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blip2-coco-captioner")

def load_model():
    if not caption_model.is_loaded:
        caption_model.load(ADAPTER_DIR)
    return caption_model

def ensure_model():
    if not caption_model.is_loaded:
        with st.spinner("🔄 Connecting to model server..."):
            load_model()


# ═══════════════════════════════════════════════════════════════
# Sidebar — History + Settings + Language Selector (NEW)
# ═══════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        # ── Logo ──
        st.markdown("""
        <div style="text-align:center; padding: 10px 0 20px;">
            <div style="font-size: 2.2rem;">🖼️</div>
            <div style="font-size: 1.1rem; font-weight: 600; margin-top: 4px;">
                <span class="gradient-text">CaptionAI</span>
            </div>
            <div style="font-size: 0.75rem; opacity: 0.5; margin-top: 2px;">
                BLIP-2 • COCO Fine-tuned
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Caption Generation Settings ──
        st.markdown("##### ⚙️ Generation Settings")
        num_beams = st.slider("Beam Width", 1, 10, 5,
                              help="Higher = more accurate and slower")
        max_tokens = st.slider("Max Caption Length", 25, 100, 50)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1,
                                help="Controls diversity of alternative captions")

        st.divider()

        # ── Language Selector (NEW) ──
        st.markdown("##### 🌐 Multilingual Captions")
        st.caption("Select languages to translate the caption into:")

        all_languages = get_language_names()

        # Show each language as a checkbox
        # Default: French, Hindi, Spanish pre-selected
        selected = []
        for lang in all_languages:
            native = LANGUAGE_NATIVE.get(lang, "")
            label = f"{lang}  `{native}`" if native else lang
            checked = lang in st.session_state.selected_languages
            if st.checkbox(label, value=checked, key=f"lang_{lang}"):
                selected.append(lang)

        # Persist selection to session state
        st.session_state.selected_languages = selected

        if not selected:
            st.caption("⚠️ No languages selected — captions will be English only.")

        st.divider()

        # ── History Panel ──
        st.markdown("##### 🕒 Recent Captions")
        history = HistoryManager.get_all(st.session_state)

        if not history:
            st.caption("No captions yet. Upload an image to start!")
        else:
            if st.button("🗑️ Clear History", use_container_width=True):
                HistoryManager.clear(st.session_state)
                st.rerun()

            for entry in history[:10]:
                col_thumb, col_text = st.columns([1, 3])
                with col_thumb:
                    st.markdown(
                        f'<img src="{entry["thumbnail"]}" '
                        f'style="width:48px;height:48px;border-radius:8px;object-fit:cover;">',
                        unsafe_allow_html=True,
                    )
                with col_text:
                    st.markdown(
                        f'<div class="caption-preview" style="font-size:0.82rem;line-height:1.35;">'
                        f'{entry["caption"][:80]}{"..." if len(entry["caption"]) > 80 else ""}'
                        f'</div>'
                        f'<div class="time-label" style="font-size:0.7rem;opacity:0.45;">'
                        f'{entry["timestamp"]}</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Load", key=f"hist_{entry['id']}", type="secondary"):
                        st.session_state.current_image = entry["image"]
                        st.rerun()

        st.divider()

        # ── Architecture Info ──
        st.markdown("##### 📐 Architecture")
        st.markdown("""
        <div style="font-size:0.8rem; line-height:1.8;">
        <span class="badge badge-blue">ViT-G/14</span>
        <span class="badge badge-purple">Q-Former</span>
        <span class="badge badge-green">OPT-2.7B</span>
        <span class="badge badge-amber">LoRA r=32</span>
        <span class="badge badge-blue">MarianMT</span>
        </div>
        """, unsafe_allow_html=True)

    return num_beams, max_tokens, temperature


# ═══════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════

def render_header():
    st.markdown("""
    <div style="text-align:center; padding: 0 0 1.5rem;">
        <h1 style="font-size: 2.4rem; font-weight: 700; margin-bottom: 6px;">
            <span class="gradient-text">Image Captioning AI</span>
        </h1>
        <p style="font-size: 1rem; opacity: 0.6; margin: 0;">
            Upload an image and get an AI-generated natural language caption
        </p>
        <div style="margin-top: 12px;">
            <span class="badge badge-purple">Encoder-Decoder</span>
            <span class="badge badge-blue">Transformer</span>
            <span class="badge badge-green">Fine-tuned on COCO</span>
            <span class="badge badge-amber">Multilingual</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Input Section
# ═══════════════════════════════════════════════════════════════

def render_input_section() -> Image.Image | None:
    if "_prev_url" not in st.session_state:
        st.session_state._prev_url = ""
    if "_prev_upload_name" not in st.session_state:
        st.session_state._prev_upload_name = None
    if "_active_source" not in st.session_state:
        st.session_state._active_source = None

    tab_upload, tab_url, tab_camera = st.tabs(["📁 Upload", "🔗 URL", "📷 Camera"])

    upload_image = url_image = cam_image = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Drag and drop or click to upload",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
            key=f"uploader_{st.session_state._uploader_key}",
        )
        if uploaded:
            upload_image = Image.open(uploaded).convert("RGB")
            if uploaded.name != st.session_state._prev_upload_name:
                st.session_state._prev_upload_name = uploaded.name
                st.session_state._active_source = "upload"

    with tab_url:
        url = st.text_input("Paste image URL", placeholder="https://example.com/photo.jpg",
                            label_visibility="collapsed")
        if url:
            try:
                url_image = load_image_from_url(url)
                if url != st.session_state._prev_url:
                    st.session_state._prev_url = url
                    st.session_state._active_source = "url"
            except ValueError as e:
                st.error(str(e))

    with tab_camera:
        cam_photo = st.camera_input("Take a photo", label_visibility="collapsed")
        if cam_photo:
            cam_image = Image.open(cam_photo).convert("RGB")
            st.session_state._active_source = "camera"

    source = st.session_state._active_source
    if source == "upload" and upload_image:
        image = upload_image
    elif source == "url" and url_image:
        image = url_image
    elif source == "camera" and cam_image:
        image = cam_image
    else:
        image = upload_image or url_image or cam_image

    if image is None and st.session_state.current_image is not None:
        image = st.session_state.current_image

    return image


# ═══════════════════════════════════════════════════════════════
# Result Panels
# ═══════════════════════════════════════════════════════════════

def render_caption_tab(result: InferenceResult):
    mc = result.main_caption
    st.markdown("#### 💬 Generated Caption")
    st.markdown(f"""
    <div style="
        font-size: 1.35rem; font-weight: 500; line-height: 1.7;
        padding: 12px 16px; border-radius: 12px;
        border-left: 4px solid #8b5cf6;
        background: rgba(139, 92, 246, 0.06);
        animation: fadeInUp 0.5s ease-out;
    ">{mc.text}</div>
    """, unsafe_allow_html=True)

    color = confidence_color(mc.confidence)
    label = confidence_label(mc.confidence)
    pct = int(mc.confidence * 100)

    st.markdown(f"""
    <div style="margin: 16px 0 8px;">
        <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
            <span style="opacity:0.6;">Confidence</span>
            <span style="font-weight:600; color:{color};">{pct}% — {label}</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if result.alternatives:
        st.markdown("#### 🔄 Alternative Captions")
        for i, alt in enumerate(result.alternatives):
            alt_color = confidence_color(alt.confidence)
            alt_pct = int(alt.confidence * 100)
            st.markdown(f"""
            <div class="alt-caption animate-in" style="animation-delay:{i * 0.1}s;">
                <span>{alt.text}</span>
                <span class="conf-badge" style="background:{alt_color}15; color:{alt_color};">{alt_pct}%</span>
            </div>
            """, unsafe_allow_html=True)


def render_attention_tab(image: Image.Image, result: InferenceResult):
    st.markdown("#### 🔥 Attention Heatmap")
    st.caption("Shows where the model focuses when generating the caption.")

    if result.attention_weights is not None:
        overlay = render_attention_overlay(image, result.attention_weights)
        col_orig, col_attn = st.columns(2)
        with col_orig:
            st.image(image, caption="Original", width="stretch")
        with col_attn:
            st.image(overlay, caption="Attention Overlay", width="stretch")
    else:
        st.info("Attention weights not available for this model configuration.")

    st.markdown("""
    <div style="font-size:0.85rem; opacity:0.6; margin-top:12px; line-height:1.6;">
        <b>How it works:</b> The Q-Former uses <b>cross-attention</b> queries to select the most
        relevant visual regions from the ViT encoder output. Brighter areas indicate higher
        attention weight.
    </div>
    """, unsafe_allow_html=True)


def render_explanation_tab(result: InferenceResult):
    st.markdown("#### 🧠 Caption Explanation")

    st.markdown("**Detected Objects**")
    tags_html = ""
    colors = ["#8b5cf6", "#3b82f6", "#06b6d4", "#22c55e", "#f59e0b", "#ef4444", "#ec4899", "#6366f1"]
    for i, obj in enumerate(result.objects_detected):
        c = colors[i % len(colors)]
        tags_html += f'<span class="obj-tag" style="color:{c}; border-color:{c}30;">{obj}</span>'
    st.markdown(f'<div style="margin-bottom:16px;">{tags_html}</div>', unsafe_allow_html=True)

    st.markdown("**Model Reasoning**")
    st.markdown(result.reasoning)

    with st.expander("📐 Architecture Pipeline", expanded=False):
        st.markdown("""
        ```
        ┌───────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────────┐
        │   Input   │────▶│ ViT-G/14     │────▶│  Q-Former    │────▶│   OPT-2.7B    │
        │   Image   │     │ (Frozen)     │     │  (Frozen)    │     │  (LoRA Tuned) │
        └───────────┘     └──────────────┘     └──────────────┘     └───────┬───────┘
                           Vision Encoder       Bridge Layer                │
                                                                    English Caption
                                                                            │
                                                                    ┌───────▼───────┐
                                                                    │  MarianMT     │
                                                                    │  (per lang)   │
                                                                    └───────┬───────┘
                                                                            │
                                                                    Multilingual Captions
        ```
        """)


def render_translations_tab(result: InferenceResult):
    """
    NEW TAB — renders multilingual translations of the caption.
    Shows a card per language with flag, language name, and translated text.
    If no languages were selected, shows a prompt to select from the sidebar.
    """
    st.markdown("#### 🌐 Multilingual Captions")

    if not result.translations:
        st.markdown("""
        <div class="translation-no-selection">
            <div style="font-size:2rem; margin-bottom:10px;">🌍</div>
            <div>No languages selected.</div>
            <div style="font-size:0.82rem; margin-top:6px; opacity:0.7;">
                Select languages in the sidebar and generate a new caption.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Summary badge row
    lang_count = len(result.translations)
    st.markdown(f"""
    <div style="margin-bottom: 16px; font-size: 0.82rem; opacity: 0.6;">
        Caption translated into <b>{lang_count}</b> language{"s" if lang_count != 1 else ""}
        using <b>Helsinki-NLP MarianMT</b> (seq2seq Transformer, OPUS corpus)
    </div>
    """, unsafe_allow_html=True)

    # Render translation cards
    cards_html = render_translation_cards(result.translations)
    st.markdown(cards_html, unsafe_allow_html=True)

    # Copy-friendly plain text expander
    with st.expander("📋 Plain text (for copying)", expanded=False):
        for lang, text in result.translations.items():
            parts = lang.split(" ", 1)
            name = parts[1] if len(parts) > 1 else lang
            st.text(f"{name}: {text}")


# ═══════════════════════════════════════════════════════════════
# Demo Section
# ═══════════════════════════════════════════════════════════════

def render_demo_section():
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; margin-bottom:20px;">
        <h3 style="margin:0;"><span class="gradient-text">Dataset Demo</span></h3>
        <p style="font-size:0.85rem; opacity:0.5;">Sample images from COCO with ground truth captions</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(DEMO_SAMPLES))
    for col, sample in zip(cols, DEMO_SAMPLES):
        with col:
            try:
                img = load_image_from_url(sample["url"])
                st.image(img, width="stretch")
            except Exception:
                st.markdown(
                    '<div style="height:200px;background:rgba(128,128,128,0.1);'
                    'border-radius:12px;display:flex;align-items:center;'
                    'justify-content:center;opacity:0.4;">Image unavailable</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(f"""
            <div style="font-size:0.82rem; margin-top:8px;">
                <div style="margin-bottom:6px;">
                    <span class="badge badge-green" style="font-size:0.68rem;">Ground Truth</span>
                </div>
                <div style="opacity:0.8; line-height:1.4;">{sample['ground_truth']}</div>
                <div style="margin-top:8px; margin-bottom:4px;">
                    <span class="badge badge-purple" style="font-size:0.68rem;">Model Prediction</span>
                </div>
                <div style="opacity:0.8; line-height:1.4;">{sample['fallback_pred']}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════

def main():
    # Load disk history on first run
    HistoryManager.load_from_disk(st.session_state)

    # Sidebar — returns generation settings
    num_beams, max_tokens, temperature = render_sidebar()

    # Ensure model is connected
    ensure_model()

    # Header
    render_header()

    # Two-column layout
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT: Image input ──
    with col_left:
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        image = render_input_section()

        if image is not None:
            st.image(image, width="stretch")
            generate = st.button("✨ Generate Caption", type="primary",
                                 use_container_width=True)
        else:
            generate = False
            st.markdown(
                '<div style="text-align:center; padding:60px 20px; opacity:0.4;">'
                '<div style="font-size:3rem; margin-bottom:8px;">🖼️</div>'
                '<div>Upload, paste a URL, or take a photo</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT: Results ──
    with col_right:
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)

        if generate and image is not None:
            loading_placeholder = st.empty()

            for step_idx in range(len(LOADING_STEPS)):
                loading_placeholder.markdown(
                    render_loading_step(step_idx), unsafe_allow_html=True
                )
                if step_idx == 0:
                    time.sleep(0.5)
                elif step_idx == 1:
                    # ── Run inference + translation ──
                    result = caption_model.generate(
                        image,
                        num_beams=num_beams,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        num_alternatives=3,
                        # Pass selected languages from sidebar to generate()
                        translate_languages=st.session_state.selected_languages,
                    )
                    st.session_state.result = result
                    st.session_state.current_image = image
                    st.session_state._uploader_key += 1
                    st.session_state._prev_upload_name = None
                    st.session_state._active_source = None

                    # Save to history — now includes translations
                    HistoryManager.add(
                        st.session_state, image,
                        result.main_caption.text,
                        result.main_caption.confidence,
                        [a.text for a in result.alternatives],
                        translations=result.translations,
                    )
                else:
                    time.sleep(0.3)

            loading_placeholder.markdown(
                render_loading_step(len(LOADING_STEPS)), unsafe_allow_html=True
            )
            time.sleep(0.2)
            loading_placeholder.empty()
            st.rerun()

        # Show results
        result = st.session_state.result
        if result is not None and image is not None:
            # ── 4 tabs now (added 🌐 Translations) ──
            tab_caption, tab_attention, tab_explain, tab_translate = st.tabs([
                "💬 Caption", "🔥 Attention Map", "🧠 Explanation", "🌐 Translations"
            ])
            with tab_caption:
                render_caption_tab(result)
            with tab_attention:
                render_attention_tab(image, result)
            with tab_explain:
                render_explanation_tab(result)
            with tab_translate:
                render_translations_tab(result)
        else:
            st.markdown(
                '<div style="text-align:center; padding:80px 20px; opacity:0.35;">'
                '<div style="font-size:2.5rem; margin-bottom:10px;">💬</div>'
                '<div style="font-size:1rem;">Caption will appear here</div>'
                '<div style="font-size:0.82rem; margin-top:6px;">Upload an image and click Generate</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Demo section
    render_demo_section()

    # Footer
    st.markdown("""
    <div style="text-align:center; padding:30px 0 10px; opacity:0.35; font-size:0.8rem;">
        DL Mini Project • BLIP-2 Image Captioning • Encoder-Decoder Transformer • Trained on COCO 2014 • Multilingual via MarianMT
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()