# utils/pdf_text_utils.py
from pathlib import Path
from PyPDF2 import PdfReader

def parse_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)
