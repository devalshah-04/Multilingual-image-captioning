"""
🖼️ BLIP-2 Image Captioning — Launcher
======================================
Just run:  python run.py
"""

import os
import sys
import subprocess

DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(DIR, "venv")

if sys.platform == "win32":
    PYTHON = os.path.join(VENV_DIR, "Scripts", "python.exe")
    PIP = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    STREAMLIT = os.path.join(VENV_DIR, "Scripts", "streamlit.exe")
else:
    PYTHON = os.path.join(VENV_DIR, "bin", "python")
    PIP = os.path.join(VENV_DIR, "bin", "pip")
    STREAMLIT = os.path.join(VENV_DIR, "bin", "streamlit")


def main():
    print("=" * 60)
    print("🖼️  BLIP-2 Image Captioning AI — Launcher")
    print("=" * 60)

    # Step 1: Create venv if missing
    if not os.path.exists(PYTHON):
        print("\n📦 Virtual environment not found. Creating...")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], cwd=DIR, check=True)
        print("✅ venv created!")

    # Step 2: Install requirements
    req_file = os.path.join(DIR, "requirements.txt")
    if os.path.exists(req_file):
        print("\n📥 Installing requirements (this may take a few minutes)...")
        subprocess.run([PIP, "install", "-q", "-r", req_file], cwd=DIR, check=True)
        print("✅ Dependencies installed!")
    else:
        print("⚠️  requirements.txt not found, skipping install.")

    # Step 3: Ask for Gradio URL
    print("\n" + "-" * 60)
    print("📡 Enter your Gradio server URL")
    print("   (Run colab_inference_server.ipynb on Kaggle to get this)")
    print("-" * 60)
    gradio_url = input("\n🔗 Gradio URL: ").strip().rstrip("/")

    if not gradio_url:
        print("❌ No URL provided. Exiting.")
        sys.exit(1)

    # Step 4: Test connection
    print(f"\n🔍 Testing connection to {gradio_url}...")
    test = subprocess.run(
        [PYTHON, "-c",
         f"import requests; r=requests.get('{gradio_url}/gradio_api/call/predict', timeout=10); "
         f"print('✅ Server is reachable!' if r.status_code in [200,405,422] else f'⚠️  Status: {{r.status_code}}')"],
        cwd=DIR, capture_output=True, text=True
    )
    print(test.stdout.strip() if test.stdout.strip() else "⚠️  Could not verify (may still work)")

    # Step 5: Launch Streamlit
    env = os.environ.copy()
    env["GRADIO_API_URL"] = gradio_url
    env["HF_TOKEN"] = env.get("HF_TOKEN", "")

    print(f"\n🚀 Starting Streamlit app...")
    print(f"   Open http://localhost:8501 in your browser")
    print(f"   Press Ctrl+C to stop\n")

    try:
        subprocess.run(
            [STREAMLIT, "run", "app.py",
             "--server.headless", "true",
             "--browser.gatherUsageStats", "false"],
            env=env, cwd=DIR
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")


if __name__ == "__main__":
    main()
