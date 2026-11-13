import re
from pathlib import Path

from .pdf_processor import extract_text_from_pdf

TARGET_CHUNK_SIZE = 1000
MAX_CHUNK_SIZE = 1300
MIN_CHUNK_SIZE = 120


def clean_text(text: str) -> str:
    text = text.replace("<EOS>", "").replace("<pad>", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_into_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def split_long_paragraph(paragraph: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
    # Keep very long single sentences bounded without splitting ordinary words.
    pieces = []
    for sentence in sentences or [paragraph]:
        if len(sentence) <= MAX_CHUNK_SIZE:
            pieces.append(sentence)
        else:
            words, current = sentence.split(), []
            size = 0
            for word in words:
                if current and size + len(word) + 1 > MAX_CHUNK_SIZE:
                    pieces.append(" ".join(current))
                    current, size = [], 0
                current.append(word)
                size += len(word) + 1
            if current:
                pieces.append(" ".join(current))
    return pieces


def build_chunks_from_paragraphs(paragraphs: list[str]) -> list[str]:
    chunks, current, length = [], [], 0
    for paragraph in paragraphs:
        pieces = split_long_paragraph(paragraph) if len(paragraph) > MAX_CHUNK_SIZE else [paragraph]
        for piece in pieces:
            if current and length + len(piece) + 2 > MAX_CHUNK_SIZE:
                chunks.append("\n\n".join(current).strip())
                current, length = [], 0
            current.append(piece)
            length += len(piece) + 2
            if length >= TARGET_CHUNK_SIZE:
                chunks.append("\n\n".join(current).strip())
                current, length = [], 0
    if current:
        chunks.append("\n\n".join(current).strip())
    return chunks


def is_useful_chunk(text: str) -> bool:
    return len(text) >= MIN_CHUNK_SIZE and sum(c.isalpha() for c in text) >= MIN_CHUNK_SIZE * .5


def chunk_page(page_text: str, page_number: int, document_name: str = "Research paper") -> list[dict]:
    text = clean_text(page_text)
    if not text:
        return []
    chunks = []
    for index, content in enumerate(build_chunks_from_paragraphs(split_into_paragraphs(text)), 1):
        if is_useful_chunk(content):
            chunks.append({"chunk_id": f"page_{page_number}_chunk_{index}", "document_name": document_name,
                           "page_number": page_number, "text": content})
    return chunks


def create_chunks(pdf_path: str | Path) -> list[dict]:
    path = Path(pdf_path)
    pages = extract_text_from_pdf(path)
    name = path.stem.replace("_", " ").replace("-", " ").title()
    return [chunk for page in pages for chunk in chunk_page(page["text"], page["page_number"], name)]
