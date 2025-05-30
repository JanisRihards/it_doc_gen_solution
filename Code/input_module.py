# input_module.py

import os
import json
import tempfile
import streamlit as st
from typing import List, Any

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(uploaded_file) -> List[Document]:
    """Read an uploaded PDF file and split it into text chunks for vector storage."""
    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    os.unlink(path)  # remove the temp file after loading
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return splitter.split_documents(docs)

def process_json(uploaded_file) -> Any:
    """Parse an uploaded JSON file into a Python object (dict or list)."""
    try:
        uploaded_file.seek(0)
        return json.load(uploaded_file)
    except json.JSONDecodeError:
        st.error(f"Failed to parse JSON: {uploaded_file.name}")
        return None
