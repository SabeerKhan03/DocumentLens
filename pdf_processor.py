from pathlib import Path
import pymupdf


def extract_text_from_pdf(pdf_path: str | Path) -> list[dict]:
    path = Path(pdf_path)
    pages = []
    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            pages.append({"page_number": page_number, "text": page.get_text("text").strip()})
    if not any(page["text"] for page in pages):
        raise ValueError("No selectable text was found. Scanned PDFs need OCR before they can be indexed.")
    return pages
