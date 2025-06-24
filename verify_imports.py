# file: verify_imports.py

import os
import sys

# Ensure root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR))
UTILS_DIR = os.path.join(ROOT_DIR, "utils")

print(f"🔍 Root: {ROOT_DIR}")
print(f"🔍 Checking 'utils/' folder exists: {os.path.isdir(UTILS_DIR)}")

sys.path.append(ROOT_DIR)

# Test critical imports
def test_import(module_path: str, label: str):
    try:
        module = __import__(module_path, fromlist=["*"])
        print(f"✅ {label} - import succeeded.")
    except Exception as e:
        print(f"❌ {label} - import failed.\n   Reason: {e}")

if __name__ == "__main__":
    test_import("utils.pdf_sales_parser", "PDF Sales Parser")
    test_import("app.streamlit_app", "Streamlit App (main)")
