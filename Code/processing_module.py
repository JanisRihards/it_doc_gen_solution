import json
import re
import streamlit as st
from typing import List

import chromadb
import ollama
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_core.documents import Document

# === System Prompt ===
system_prompt = """
You are a document generation assistant local model llama3.2:3b.
You will receive:
  -<Context>: You will receive text fragments for context input
  -<Instructions>: JSON payload with processing rules under 'llm_prompt_instructions'.
  -<Output structure>: JSON payload defining 'output_structure'.
  -<User input>: optional additional information.
Tasks:
  1. Process <Context> according to <Instructions> JSON.
  2. Generate output strictly following <Output structure> JSON keys and hierarchy.
  3. If <Context> lacks any required field, insert a placeholder in the form <MISSING_field_name>.
  4. Do not add information not present in <Context>:.

Master rule: If you do not find information based on specific <Instructions>:. Leave output empty.
"""

def get_vector_collection() -> chromadb.Collection:
    """Initialize or retrieve the persistent ChromaDB collection for the app."""
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

def upsert_items(texts: List[str], metadatas: List[dict], ids: List[str]) -> bool:
    """Insert a batch of documents with metadata and IDs into the ChromaDB collection."""
    if texts:
        col = get_vector_collection()
        col.upsert(documents=texts, metadatas=metadatas, ids=ids)
        return True
    return False

def add_documents_to_vectorstore(docs: List[Document], source_name: str, doc_type: str):
    """Upsert chunks of a PDF document into the vector store with the given type (context or instruction)."""
    texts, metas, ids = [], [], []
    for i, d in enumerate(docs):
        txt = d.page_content.strip()
        if not txt:
            continue
        texts.append(txt)
        metas.append({"type": doc_type, "source": source_name})
        ids.append(f"{source_name}_{doc_type}_{i}")
    if upsert_items(texts, metas, ids):
        st.success(f"Uploaded {doc_type} from {source_name}.")
    else:
        st.warning(f"No content found in {source_name} for type {doc_type}.")

def call_llm(context: str, instructions: str, user_input: str, example: str):
    """
    Call the local LLaMA 3.2:3b model via Ollama with the given context, instructions, user input, and example structure.
    Streams the response and yields chunks of text.
    """
    prompt = (
        f"Context:\n{context}\n\n"
        f"Instructions:\n{instructions}\n\n"
        f"User_input:\n{user_input}\n\n"
        f"Example_structure:\n{example}"
    )
    resp = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
    )
    for chunk in resp:
        if not chunk.get("done", False):
            yield chunk["message"]["content"]
