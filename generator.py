import os

from dotenv import load_dotenv
from google import genai

from .prompts import SYSTEM_INSTRUCTION
from .retriever import Retriever


def generate_answer(question: str, retriever: Retriever | None = None, top_k: int = 5) -> tuple[str, list[dict]]:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Add GEMINI_API_KEY to your .env file to enable answers.")
    retriever = retriever or Retriever("chroma_db")
    results = retriever.retrieve(question, top_k=top_k)
    if not results:
        return "No indexed paper passages were found. Add and index a PDF first.", []
    context = "\n\n".join(
        f"[Source: {item['document_name']}, page {item['page_number']}]\n{item['text']}" for item in results
    )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=f"{SYSTEM_INSTRUCTION}\n\nPaper excerpts:\n{context}\n\nQuestion: {question}",
    )
    return (response.text or "The model returned an empty response."), results
