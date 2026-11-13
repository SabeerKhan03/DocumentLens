from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv

from src.generator import generate_answer
from src.retriever import Retriever
from src.vector_store import index_pdf, get_collection

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "papers"
DB_DIR = ROOT / "chroma_db"
DATA_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="DocumentLens", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1120px; padding-top: 2.2rem;}
[data-testid="stSidebar"] {background: #f5f7fa;}
.source-card {padding: 1rem 1.1rem; border: 1px solid #e5e9f0; border-radius: 12px; margin: .6rem 0; background: white;}
</style>""", unsafe_allow_html=True)

st.title("🔎 DocumentLens")
st.caption("Ask questions about your research papers. Answers are grounded in indexed passages and include page citations.")

with st.sidebar:
    st.header("Your library")
    uploaded = st.file_uploader("Add a research paper", type=["pdf"], help="Text based PDFs are supported; scanned documents need OCR first.")
    if uploaded and st.button("Save and index paper", type="primary", use_container_width=True):
        target = DATA_DIR / Path(uploaded.name).name
        target.write_bytes(uploaded.getvalue())
        try:
            with st.spinner("Reading pages and building search index…"):
                total = index_pdf(target, DB_DIR)
            st.success(f"Indexed {total} passages from {target.name}.")
        except Exception as exc:
            st.error(f"Could not index this PDF: {exc}")
    if st.button("Index included sample paper", use_container_width=True):
        sample = DATA_DIR / "attention_is_all_you_need.pdf"
        if sample.exists():
            try:
                with st.spinner("Building search index…"):
                    total = index_pdf(sample, DB_DIR)
                st.success(f"Indexed {total} passages from the sample paper.")
            except Exception as exc:
                st.error(f"Could not index the sample: {exc}")
        else:
            st.error("The sample paper is missing from data/papers.")
    st.divider()
    try:
        indexed_count = get_collection(DB_DIR).count()
    except Exception:
        indexed_count = 0
    st.metric("Indexed passages", indexed_count)
    if not os.getenv("GEMINI_API_KEY"):
        st.info("Add a Gemini API key in .env to enable question answering.")

st.subheader("Ask your library")
question = st.text_area("Question", placeholder="For example: Why does the Transformer use positional encodings?", height=100, label_visibility="collapsed")
if st.button("Find an answer", type="primary", disabled=not question.strip()):
    if indexed_count == 0:
        st.warning("Index a paper from the sidebar first.")
    elif not os.getenv("GEMINI_API_KEY"):
        st.warning("Set GEMINI_API_KEY in the project’s .env file, then restart the app.")
    else:
        try:
            with st.spinner("Searching your papers and writing a cited answer…"):
                retriever = Retriever(DB_DIR)
                answer, sources = generate_answer(question.strip(), retriever, top_k=6)
            st.markdown("### Answer")
            st.markdown(answer)
            if sources:
                st.markdown("### Retrieved passages")
                for source in sources:
                    st.markdown(f"**{source['document_name']} · page {source['page_number']}**  ")
                    st.caption(source["text"])
        except Exception as exc:
            st.error(f"Could not generate an answer: {exc}")

st.divider()
st.caption("DocumentLens can miss relevant passages. Check cited pages in the original paper before relying on an answer.")
