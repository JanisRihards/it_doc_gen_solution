## Overview

##What is this prototype??##  
- A local web app (built with Streamlit) that implements a simple RAG pipeline.  
- You upload one or more PDFs (the “context”) and two JSON files (one containing LLM prompt/instruction definitions, and one defining the desired output structure).  
- The app:
  1. Splits each PDF into small text chunks (using PyMuPDF + a text splitter).  
  2. Automatically embeds those chunks with a local Ollama embedding model and indexes them in ChromaDB.  
  3. For each “step” in your JSON instructions, it queries ChromaDB (use Ollama embeddings under the hood) to retrieve the top-k relevant chunks.  
  4. Feeds the retrieved context, plus instructions and any user prompt, into LLaMA (3.2:3b) via Ollama’s chat API.  
  5. Parses the LLM’s streamed JSON response, upserts it back into ChromaDB (for traceability), and merges partial results step by step.  
  6. Finally, formats and displays the assembled JSON artefact in the Streamlit UI.

This prototype runs entirely on  local machine. All embeddings and LLM inference happen via a locally hosted Ollama server (current localhost point `http://localhost:11434`), and ChromaDB stores vectors on disk (`./demo-rag-chroma` by default).

---

## Prerequisites

1. Python 3.8+ installed on your system.  
2. Ollama installed locally, with the embedding (nomic-embed-text:latest) and LLaMA 3.2:3b models already pulled.  
     ollama pull nomic-embed-text:latest
     ollama pull llama3.2:3b
3. A GPU (with ≥ 4 GB VRAM) or a capable CPU if you plan to run LLaMA 3.2:3b quantized.  

---

## Local Installation

1. Clone this repository

   git clone https://github.com/JanisRihards/it_doc_gen_solution.git

   cd it_doc_gen_solution
3. Setup
 
 Run in local terminal pip install -r python_libraries_install.txt
   
