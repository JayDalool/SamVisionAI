
# file: run_all_pipeline.py

import os
import subprocess
import datetime

def run_script(path, desc):
    print(f"🚀 Running: {desc}")
    result = subprocess.run(["python", path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Success: {desc}\n")
    else:
        print(f"❌ Failed: {desc}")
        print(result.stderr)

if __name__ == "__main__":
    steps = [
        #("data/generate_real_estate_data.py", "Generate Synthetic Real Estate Data"),
        ("data/load_to_postgresql.py", "Load Data to PostgreSQL"),
        ("ml/train_model.py", "Train Prediction Model")
    ]

    print(f"🕒 Started pipeline at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    for script_path, description in steps:
        run_script(script_path, description)

    print("🎉 All steps completed! You can now run `streamlit run app/streamlit_app.py`.")
