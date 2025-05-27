# file: run_all_pipeline.py

import os
import subprocess
import datetime

def run_script(path, desc, module=False):
    print(f"🚀 Running: {desc}")
    if module:
        path = path.replace("/", ".").replace(".py", "")
        result = subprocess.run(["python", "-m", path], capture_output=True, text=True)
    else:
        result = subprocess.run(["python", path], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Success: {desc}\n")
    else:
        print(f"❌ Failed: {desc}")
        print(result.stderr)

if __name__ == "__main__":
    steps = [
        ("data.merge_missing_data", "Merge Missing Addresses", True),
        ("data.load_to_postgresql", "Load Real MLS Data to PostgreSQL", True),
        ("ml/train_model.py", "Train Prediction Model", False)
    ]

    print(f"🕒 Started pipeline at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    for script_path, description, as_module in steps:
        run_script(script_path, description, module=as_module)

    print("🎉 All steps completed! You can now run `streamlit run app/streamlit_app.py`.")
