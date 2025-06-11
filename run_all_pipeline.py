# file: run_all_pipeline.py

import os
import subprocess
import datetime
import shutil


def run_script(path, desc, module=False):
    print(f"\n🚀 Running: {desc}")
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
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"🕒 Started pipeline at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Backup existing housing data CSV
    if os.path.exists("winnipeg_housing_data.csv"):
        os.makedirs("backup", exist_ok=True)
        backup_path = f"backup/winnipeg_housing_data_{timestamp}.csv"
        shutil.copy("winnipeg_housing_data.csv", backup_path)
        print(f"📦 CSV backup saved to: {backup_path}")

    # Execute all scripts in order
    steps = [
        ("tools/pdf_to_csv.py", "Step 1: Parse PDF files into CSV", False),
        ("data/clean_and_validate_csv.py", "Step 2: Clean & Validate Parsed CSV", False),
        ("data/load_to_postgresql.py", "Step 3: Insert Cleaned Data to DB", False),
        ("ml/train_model.py", "Step 4: Train Prediction Model", False),
    ]

    for script_path, description, as_module in steps:
        run_script(script_path, description, module=as_module)

    print("\n🎉 Pipeline complete!")
    print("💡 Run this to launch: `streamlit run app/streamlit_app.py`")
