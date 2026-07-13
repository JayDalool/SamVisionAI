"""Coordinate-aware PDF text extraction (text layer first).

These WRREB Matrix exports carry a clean text layer, so we extract positioned
text lines with pdfminer and reconstruct rows/columns geometrically instead of
regexing a flattened blob. OCR is intentionally NOT performed here: if a page's
text layer is empty or unusably thin we FLAG it (status="needs_ocr") so nothing
critical is ever silently guessed. Real OCR is a future fallback, not an MVP
dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTTextLineHorizontal

# Below this many characters a page's text layer is considered missing/corrupt.
MIN_PAGE_CHARS = 40


@dataclass
class TextFragment:
    """A positioned line of text. y grows upward (PDF coords); x0 is the left."""
    y: float
    x0: float
    x1: float
    text: str


@dataclass
class PageLayout:
    page_number: int          # 1-indexed
    fragments: list[TextFragment]

    @property
    def char_count(self) -> int:
        return sum(len(f.text) for f in self.fragments)

    @property
    def is_thin(self) -> bool:
        return self.char_count < MIN_PAGE_CHARS

    def flat_text(self) -> str:
        # top-to-bottom, left-to-right reading order
        ordered = sorted(self.fragments, key=lambda f: (-f.y, f.x0))
        return "\n".join(f.text for f in ordered)

    def rows(self, y_tol: float = 2.5) -> list[list[TextFragment]]:
        """Group fragments that share a baseline into visual rows (top-first)."""
        rows: list[list[TextFragment]] = []
        for frag in sorted(self.fragments, key=lambda f: (-f.y, f.x0)):
            for row in rows:
                if abs(row[0].y - frag.y) <= y_tol:
                    row.append(frag)
                    break
            else:
                rows.append([frag])
        for row in rows:
            row.sort(key=lambda f: f.x0)
        return rows


def iter_pages(path: str, laparams: LAParams | None = None) -> Iterator[PageLayout]:
    laparams = laparams or LAParams()
    for i, page in enumerate(extract_pages(path, laparams=laparams), start=1):
        frags: list[TextFragment] = []
        for element in page:
            if isinstance(element, LTTextContainer):
                for line in element:
                    if isinstance(line, LTTextLineHorizontal):
                        txt = line.get_text().strip()
                        if txt:
                            frags.append(TextFragment(round(line.y0, 1), round(line.x0, 1),
                                                      round(line.x1, 1), txt))
        yield PageLayout(page_number=i, fragments=frags)
