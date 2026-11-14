# DocumentLens

DocumentLens is a local web app for asking questions about research papers. It extracts text from PDFs, splits it into page-aware passages, finds relevant passages with semantic search, and uses Gemini to write an answer with document and page citations.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key for generated answers
- Internet access on first launch so Sentence Transformers can download its embedding model

## Run on Windows

Open PowerShell in this folder and run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
streamlit run app.py
```

Put your Gemini API key in `.env`, save, and return to the app. The key stays on your computer. You may also set `GEMINI_MODEL`; the default is `gemini-2.5-flash`.

## Use it

1. Choose **Index included sample paper** or upload a PDF and choose **Save and index paper**.
2. Enter a question and select **Find an answer**.
3. Check the cited page and retrieved passages against the source PDF.

The vector index is stored locally in `chroma_db/`; uploaded PDFs are stored in `data/papers/`. Re-index a changed PDF to refresh its passages. Scanned PDFs without selectable text need OCR before indexing. The first indexing run downloads the `all-MiniLM-L6-v2` embedding model; generated answers are sent to the Gemini API along with retrieved excerpts.

## Project layout

- `app.py` — Streamlit interface
- `src/pdf_processor.py`, `src/chunker.py` — PDF text extraction and page-aware chunking
- `src/embeddings.py`, `src/vector_store.py`, `src/retriever.py` — local semantic search
- `src/generator.py`, `src/prompts.py` — grounded Gemini answers
- `data/papers/` — included sample and user-added PDFs
