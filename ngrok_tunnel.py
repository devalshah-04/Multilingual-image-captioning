"""
ngrok tunnel for Streamlit app.
Keep this script running to maintain the public URL.
"""
from pyngrok import ngrok
import time

tunnel = ngrok.connect(8501, "http")
print(f"\n{'='*60}")
print(f"🌐 PUBLIC URL: {tunnel.public_url}")
print(f"{'='*60}")
print(f"\nShare this URL to access your Image Captioning AI!")
print(f"Press Ctrl+C to stop the tunnel.\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down tunnel...")
    ngrok.kill()
