import requests, json
from PIL import Image
import io

url = "https://aca13fe307f94841ce.gradio.live"

# Create test image
img = Image.new("RGB", (224, 224), color=(255, 0, 0))
buf = io.BytesIO()
img.save(buf, format="PNG")
buf.seek(0)

# Step 1: Upload
print("Uploading...")
upload_resp = requests.post(f"{url}/gradio_api/upload", files={"files": ("test.png", buf, "image/png")}, timeout=30)
file_path = upload_resp.json()[0]
print(f"Uploaded: {file_path}")

# Step 2: Submit with ImageData dict
image_data = {"path": file_path, "orig_name": "test.png", "mime_type": "image/png", "meta": {"_type": "gradio.FileData"}}
print(f"Sending: {json.dumps(image_data)}")
resp = requests.post(f"{url}/gradio_api/call/predict", json={"data": [image_data]}, timeout=30)
print(f"Submit: {resp.status_code} -> {resp.text}")
event_id = resp.json().get("event_id")

# Step 3: Get result
result = requests.get(f"{url}/gradio_api/call/predict/{event_id}", timeout=120, stream=True)
for line in result.iter_lines(decode_unicode=True):
    print(f"LINE: [{line}]")
