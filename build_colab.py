import json

nb = {"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"provenance":[],"gpuType":"T4"},"kernelspec":{"name":"python3","display_name":"Python 3"},"accelerator":"GPU"},"cells":[]}

def cell(ct, src):
    lines = [l+"\n" for l in src.split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    c = {"cell_type":ct,"metadata":{},"source":lines}
    if ct == "code":
        c["outputs"] = []
        c["execution_count"] = None
    nb["cells"].append(c)

cell("markdown", """# 🖼️ BLIP-2 Caption Server (Fine-tuned on COCO)

**Run this on Google Colab with a T4 GPU (free).**
It loads your fine-tuned BLIP-2 OPT-6.7B + LoRA and creates a public Gradio URL.
Your local Streamlit app connects to this URL.""")

cell("code", """!pip install -q transformers accelerate peft gradio pillow torch""")

cell("code", """import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from peft import PeftModel
from PIL import Image
import gradio as gr

ADAPTER_REPO = "Parthg0106/DL-mini"
BASE_MODEL = "Salesforce/blip2-opt-6.7b"

print("Loading processor...")
processor = Blip2Processor.from_pretrained(ADAPTER_REPO)

print(f"Loading {BASE_MODEL}...")
model = Blip2ForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
)

print("Applying LoRA weights...")
model = PeftModel.from_pretrained(model, ADAPTER_REPO)
model.eval()

print(f"✅ Model loaded on {model.device}")
print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")""")

cell("code", """def caption(image):
    if image is None:
        return "Upload an image."
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pv = inputs["pixel_values"].to(model.device, dtype=torch.float16)
    with torch.no_grad():
        out = model.generate(pixel_values=pv, max_new_tokens=50, num_beams=5, early_stopping=True)
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()

demo = gr.Interface(
    fn=caption,
    inputs=gr.Image(label="Upload Image"),
    outputs=gr.Textbox(label="Caption"),
    title="🖼️ BLIP-2 Image Captioner (Fine-tuned)",
    description="BLIP-2 OPT-6.7B + LoRA fine-tuned on MS-COCO (BLEU-4: 0.42)",
)

demo.launch(share=True, debug=True)""")

cell("markdown", """## 📋 How to use

1. Run all cells above
2. Copy the **public URL** (e.g. `https://xxxxx.gradio.live`)
3. On your local PC, set the environment variable:
   ```
   set GRADIO_API_URL=https://xxxxx.gradio.live
   ```
4. Run your local Streamlit app: `streamlit run app.py`

The Gradio URL is valid for **72 hours**. Keep this Colab tab open.""")

path = r"c:\Users\Admin\Desktop\DL mini project\colab_inference_server.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False)

import os
print(f"Created: {path} ({os.path.getsize(path)} bytes)")
