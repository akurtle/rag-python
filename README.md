# RAG QA Assistant — Retrieval-Augmented Generation with Quality Evaluation

This project is a **Retrieval-Augmented Generation (RAG)** system built in Python that lets you **chat over local documentation** (TXT/PDF).  
It goes one step further and includes **Quality Engineering evaluation hooks** to measure retrieval relevance, check grounding to reduce hallucinations, and detect answer drift over time.

This makes it useful not just as a chatbot — **but as a system that can be tested, validated, and monitored**, which is essential in quality engineering roles.

---

## What the System Does

| Component | Purpose |
|----------|---------|
| **ingestion pipeline** | loads & chunks documentation, generates embeddings, stores vectors |
| **vector database (Chroma)** | enables semantic similarity search over documents |
| **RAG chatbot** | answers questions using retrieved chunks — grounded in your docs |
| **retrieval evaluation** | measures how relevant retrieved chunks are |
| **grounding evaluation** | detects hallucinations by comparing answer vs context |
| **regression testing** | detects semantic drift if answers change across model versions or prompt changes |

---

## Tech Overview

- **Python**
- **OpenAI embeddings** (`text-embedding-3-small`)
- **ChromaDB** (persistent vector store)
- **RAG pipeline** (retrieval → grounding → generation)
- **Evaluation utilities** for QE-style validation

---

## Project Structure


## Quick Start

### Install

```bash
pip install -r requirements.txt
```

Add your API key to .env:

```ini
OPENAI_API_KEY=your_key_here
```

``` bash 
python -m src.ingest
```


This will:

read files in data/
chunk them with overlap
embed chunks
store vectors in ChromaDB



``` bash 
python -m src.rag_chat

```
