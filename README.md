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

---

## Testing & Quality Evaluation

Run the full test suite with:

```bash
pytest -v
```

The suite covers:

| Test file | Purpose |
|-----------|---------|
| `tests/test_retrieval.py` | Retrieval relevance — checks that retrieved chunks contain expected keywords for known questions |
| `tests/test_grounding.py` | Hallucination detection — for out-of-scope questions, the model must admit it doesn't know rather than making things up |
| `tests/test_hallucination_and_relevance.py` | Quantitative grounding score (% of answer tokens found in context), answer correctness for in-scope questions (expected facts present in the answer), and embedding-based cosine similarity between queries and retrieved chunks |
| `tests/test_regression.py` | Semantic regression — flags drift if answers diverge too far from expected reference answers |
| `tests/tests_robustness.py` | Fuzz testing — ensures the system handles empty, malformed, or adversarial queries without crashing |

A session-scoped fixture (`tests/conftest.py`) automatically runs ingestion if `chroma_db/` is missing or empty, so the suite can run against a fresh checkout.
