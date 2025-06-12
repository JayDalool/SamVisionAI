import subprocess
import os
import datetime
import sys

def run_step(label, command):
    print(f"\n🚀 Running: {label}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {label}")
        sys.exit(1)
    print(f"✅ Success: {label}")

if __name__ == '__main__':
    print(f"🕒 Started pipeline at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    run_step("Step 1: Parse PDF files into CSV", "python tools/pdf_to_csv.py")
    run_step("Step 2: Clean & Validate Parsed CSV", "python data/clean_and_validate_csv.py")
    run_step("Step 3: Insert Cleaned Data to DB", "python data/load_to_postgresql.py")
    run_step("Step 4: Train Prediction Model", "python ml/train_model.py")

    print("\n🎉 Pipeline complete!")
    print("💡 Run this to launch: `streamlit run app/streamlit_app.py`")
