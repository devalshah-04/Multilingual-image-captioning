# 🖼️ Multilingual Image Captioning AI

> **BLIP-2 fine-tuned on MS-COCO + Helsinki-NLP MarianMT Neural Machine Translation — deployed on Streamlit Cloud + AWS**

A deep learning pipeline that generates natural language captions for any image and translates them into 10 languages in real time. Built combining **Computer Vision**, **Multimodal AI**, and **Neural Machine Translation**, with a cloud backend on **AWS**.

---

## 🔗 Live Demo & Endpoints

| Service | URL |
|---|---|
| 🚀 Live App | [deval-multilingual-image-captioning.streamlit.app](https://deval-multilingual-image-captioning.streamlit.app/) |
| ☁️ AWS API Gateway | `https://6lnxnivdxf.execute-api.ap-south-1.amazonaws.com/prod/caption` *(hosted, currently inactive)* |
| 🤗 BLIP-2 LoRA Adapter | [devalshah04/blip2-coco-multilingual-image-captioning](https://huggingface.co/devalshah04/blip2-coco-multilingual-image-captioning) |

---

## 📸 What It Does

1. **Upload any image** — via file upload, URL, or camera
2. **BLIP-2 generates an English caption** — via HuggingFace Inference API (always on) or fine-tuned LoRA model (requires Kaggle GPU session)
3. **MarianMT translates the caption** — into up to 10 languages
4. **Results displayed** — with confidence score, alternatives, attention heatmap, and explanation
5. **Auto-categorized and saved** — to persistent history with 15 category labels

---

## ⚡ Deployment Modes

This app runs in two modes depending on whether a Kaggle GPU session is active:

### Mode 1 — Always-On (Default, No Setup Required)
The live Streamlit app runs 24/7 using the **HuggingFace Inference API** as the caption backend.

| Feature | Status |
|---|---|
| Image upload | ✅ Always works |
| Caption generation | ✅ Always works (base BLIP-2) |
| Alternative captions | ✅ Always works |
| Attention heatmap | ✅ Always works |
| Multilingual translations (10 languages) | ✅ Always works |
| History + auto-categorization | ✅ Always works |
| **LoRA fine-tuned captions (BLEU-4: 0.42)** | ⚡ Requires Kaggle session (see below) |

### Mode 2 — LoRA Fine-tuned Captions (Higher Quality)
To enable the fine-tuned BLIP-2 model (trained on MS-COCO with LoRA), a fresh Gradio URL from a Kaggle GPU session must be set in Streamlit secrets.

**Steps to enable LoRA mode:**

1. Go to [kaggle.com](https://kaggle.com) → Open `colab_inference_server.ipynb`
2. Enable **GPU T4** under Session Options → Run all cells (~5 min)
3. Copy the public Gradio URL from the output:
   ```
   Running on public URL: https://xxxxxxxxxxxxxx.gradio.live
   ```
4. Go to your Streamlit app → **Manage app** (bottom right) → **Settings** → **Secrets**
5. Update the secret:
   ```toml
   GRADIO_API_URL = "https://xxxxxxxxxxxxxx.gradio.live"
   ```
6. Save → app restarts and switches to fine-tuned mode automatically

> **Note:** Kaggle Gradio URLs expire when the session ends (max 12 hours). When expired, the app automatically falls back to HuggingFace API — captions still generate, just without LoRA fine-tuning.

---

## 🧠 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                               │
│          deval-multilingual-image-captioning.streamlit.app          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Upload Image
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STREAMLIT CLOUD (app.py)                               │
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────────┐   │
│  │  Image      │    │ model_utils  │    │   multilingual.py     │   │
│  │  Input      │───▶│ .py          │    │                       │  │
│  │  (PIL)      │    │              │    │  MarianMT en→fr       │   │
│  └─────────────┘    │  Gradio API  │    │  MarianMT en→hi       │   │
│                     │  or HF API   │    │  MarianMT en→es       │   │
│  ┌─────────────┐    │  (fallback)  │    │  MarianMT en→de  ...  │   │
│  │  utils.py   │    └──────┬───────┘    └───────────┬───────────┘   │
│  │  History    │           │                        │               │
│  │  Category   │           ▼                        ▼               │
│  │  CSS/UI     │    English Caption         Multilingual Captions   │
│  └─────────────┘           │                        │               │
│                             └───────────┬────────────┘              │
│                                         ▼                           │
│                              ┌─────────────────────┐                │
│                              │   Streamlit UI      │                │
│                              │  💬 Caption Tab     │               │
│                              │  🔥 Attention Tab   │               │
│                              │  🧠 Explanation Tab │               │
│                              │  🌐 Translations Tab│               │
│                              └─────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌──────────────────────┐    ┌─────────────────────────┐
│   KAGGLE GPU         │    │      AWS CLOUD           │
│   (optional)         │    │                          │
│                      │    │  S3 Bucket               │
│  BLIP-2 OPT-6.7B    │    │  multilingual-captioning  │
│  + ViT-G/14         │    │  -bucket (ap-south-1)     │
│  + Q-Former         │    │          ↓                │
│  + LoRA Adapter     │    │  Lambda Function          │
│                      │    │  captioning-pipeline     │
│  Gradio Server       │    │          ↓               │
│  (fresh URL needed)  │    │  API Gateway             │
└──────────────────────┘    │  /caption POST           │
          ↑                  └─────────────────────────┘
    Only needed for
    LoRA fine-tuned
    captions
```

---

## 🤖 AI Models Used

### 1. BLIP-2 (Image Captioning)
| Component | Detail |
|---|---|
| Base Model | `Salesforce/blip2-opt-6.7b` |
| Vision Encoder | ViT-G/14 — 1.8B parameters, processes image as 257 patches |
| Bridge | Q-Former — 32 cross-attention queries connecting vision to language |
| Language Decoder | OPT-6.7B — autoregressive text generation |
| Fine-tuning | LoRA (r=32) on MS-COCO 2014 — 120K images |
| Adapter | `Parthg0106/DL-mini` on HuggingFace (377MB) |
| BLEU-4 Score | 0.42 on COCO validation set |

### 2. MarianMT (Neural Machine Translation)
| Component | Detail |
|---|---|
| Architecture | Transformer Encoder-Decoder (Vaswani et al., 2017) |
| Parameters | ~70M per language pair model |
| Training Data | OPUS Corpus — ~14 billion parallel sentence pairs |
| Languages | 10 (French, Spanish, German, Hindi, Japanese, Chinese, Portuguese, Russian, Italian, Arabic) |
| Inference | Beam search (4 beams), runs on Streamlit Cloud CPU |
| Models | Helsinki-NLP on HuggingFace (~300MB each, cached) |

---

## ☁️ AWS Cloud Architecture

This project uses AWS for **persistent image storage**, **serverless pipeline orchestration**, and **REST API exposure**.

### Services Used

| AWS Service | Purpose | Details |
|---|---|---|
| **S3** | Image storage | Bucket: `multilingual-captioning-bucket`, Region: `ap-south-1` (Mumbai) |
| **Lambda** | Serverless compute | Function: `captioning-pipeline`, Runtime: Python 3.11 |
| **API Gateway** | REST API | Endpoint: `/caption` POST, Stage: `prod` |
| **IAM** | Access control | Role: `lambda-captioning-role` with S3 + Lambda permissions |

### AWS Data Flow

```
POST /caption (image + metadata)
        ↓
API Gateway (REST API — prod stage)
        ↓
Lambda Function (captioning-pipeline)
        ↓
  ┌─────────────────────────────┐
  │ 1. Receive image payload    │
  │ 2. Save to S3 bucket        │
  │ 3. Log caption metadata     │
  │ 4. Return S3 object URL     │
  └─────────────────────────────┘
        ↓
S3 Bucket (multilingual-captioning-bucket, ap-south-1)
```

### Why AWS?

- BLIP-2 requires **~8GB VRAM** — impossible on standard laptops
- S3 provides **persistent, scalable storage** for captioned images across sessions
- Lambda + API Gateway provides a **serverless, always-on REST interface** decoupled from the frontend
- Cloud deployment means **zero local GPU requirement** for end users

---

## 🗂️ Project Structure

```
dl-mini-project/
│
├── app.py                        # Streamlit frontend — main UI
├── model_utils.py                # BLIP-2 inference wrapper + Gradio/HF API client
├── multilingual.py               # MarianMT NMT module — 10 language translation
├── utils.py                      # UI helpers, CSS, history manager, categorizer
├── aws_integration.py            # AWS S3 + API Gateway integration (boto3)
├── evaluate_translations.py      # BLEU + chrF evaluation script
├── run.py                        # One-click local launcher
├── requirements.txt              # Python dependencies
│
├── caption_history/              # Persistent local history
│   ├── index.json                # Master log of all captions + translations
│   ├── Human/                   # Auto-categorized image folders
│   ├── Animal/
│   ├── Technology/
│   ├── Indoor/
│   └── ...
│
├── hf_space/                     # HuggingFace Spaces files
│   ├── app.py                    # Gradio interface
│   └── requirements.txt
│
├── colab_inference_server.ipynb  # GPU inference server (run on Kaggle)
├── architecture_diagram.png      # System architecture diagram
└── block_diagram.png             # Block diagram
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.10+
- Kaggle account (for LoRA fine-tuned captions — optional)

### Step 1 — (Optional) Start Kaggle GPU Server for LoRA captions
1. Open `colab_inference_server.ipynb` on [kaggle.com](https://kaggle.com)
2. Enable **GPU T4** → Run all cells (~5 min)
3. Copy the public Gradio URL from output
4. Skip this step to use HuggingFace API fallback instead

### Step 2 — Launch the App
```bash
python run.py
```
- Creates virtual environment automatically
- Installs all dependencies
- Prompts for Gradio URL (press Enter to skip → uses HF API)
- Launches at `http://localhost:8501`

---

## 📊 Translation Evaluation (BLEU Scores)

Evaluated on 8 MS-COCO style captions per language with human reference translations:

| Language | BLEU Score | chrF Score | Quality |
|---|---|---|---|
| French | ~50 | ~65 | Excellent |
| Spanish | ~48 | ~63 | Excellent |
| Italian | ~47 | ~62 | Good |
| German | ~42 | ~58 | Good |
| Portuguese | ~44 | ~60 | Good |
| Russian | ~35 | ~52 | Good |
| Hindi | ~28 | ~45 | Fair* |
| Arabic | ~30 | ~48 | Fair* |
| Chinese | ~25 | ~42 | Fair* |
| Japanese | ~22 | ~40 | Fair* |

*Lower scores for Hindi/Japanese/Arabic/Chinese reflect fundamentally different grammar structures from English and significantly less OPUS training data — a known BLEU limitation for distant language pairs.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit (deployed on Streamlit Community Cloud) |
| Image Captioning | BLIP-2 (Salesforce), LoRA fine-tuning (PEFT) |
| Translation | Helsinki-NLP MarianMT (HuggingFace Transformers) |
| Cloud Storage | AWS S3 (boto3) |
| Cloud Compute | AWS Lambda + API Gateway |
| GPU Server | Kaggle (T4 GPU, optional) |
| Model Hub | HuggingFace Hub |
| Evaluation | sacrebleu (BLEU + chrF) |

---

## 📚 References

- Li, J. et al. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.* ICML 2023.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- Helsinki-NLP. *OPUS-MT: Open translation models.* HuggingFace Hub.
- Tiedemann, J. (2012). *Parallel Data, Tools and Interfaces in OPUS.* LREC 2012.
- Papineni, K. et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation.* ACL 2002.

---

## 📄 License
MIT License — feel free to use, modify, and distribute.
