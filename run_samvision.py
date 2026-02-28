# file: run_samvision.py
import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

# Make project root importable so we can import data.load_to_postgresql
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ✅ DB bootstrap (creates DB + table + loads data)
try:
    from data.load_to_postgresql import main as bootstrap_db
except Exception as e:
    print(f"[WARN] Could not import DB bootstrap: {e}")
    bootstrap_db = None


def init_db():
    if not bootstrap_db:
        return
    try:
        print("[INFO] Initializing / refreshing local SamVision database...")
        bootstrap_db()  # safe to run multiple times (ON CONFLICT DO NOTHING)
        print("[INFO] DB initialization finished.")
    except Exception as e:
        print(f"[ERROR] DB initialization failed: {e}")


def start_streamlit():
    # Use the current Python (in venv) to run Streamlit
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--server.headless=false",
    ]
    subprocess.Popen(cmd)
    time.sleep(2)  # give Streamlit a moment to start
    webbrowser.open("http://127.0.0.1:8501")


if __name__ == "__main__":
    init_db()
    start_streamlit()
