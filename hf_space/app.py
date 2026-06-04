#hf_space/app.py

import gradio as gr
import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from peft import PeftModel
from PIL import Image
import spaces

# ── Load model at startup ──
BASE_MODEL = "Salesforce/blip2-opt-6.7b"
ADAPTER_REPO = "devalshah04/blip2-coco-multilingual-image-captioning"

print("Loading processor...")
processor = Blip2Processor.from_pretrained(ADAPTER_REPO)

print(f"Loading {BASE_MODEL}...")
model = Blip2ForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
)

print("Applying LoRA adapters...")
model = PeftModel.from_pretrained(model, ADAPTER_REPO)
model.eval()
print("Model ready!")


@spaces.GPU
def generate_caption(image: Image.Image) -> str:
    """Generate caption for an image using fine-tuned BLIP-2."""
    if image is None:
        return "Please upload an image."
    
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(model.device, dtype=torch.float16)

    with torch.no_grad():
        output = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=50,
            num_beams=5,
            early_stopping=True,
        )
    
    caption = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
    return caption


# ── Gradio UI ──
demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="🖼️ BLIP-2 Image Captioning (Fine-tuned on COCO)",
    description="BLIP-2 OPT-6.7B fine-tuned with LoRA on MS-COCO. Upload an image to generate a caption.",
    examples=[],
    allow_flagging="never",
)

demo.launch()
