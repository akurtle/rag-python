import os
import textwrap
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPEN_API_KEY")

    if not api_key:
        raise ValueError("Open API key error")
    
    return OpenAI(api_key=api_key)



def chunk_text(
    text: str,
    max_tokens: int = 300,
    overlap: int = 50,
) -> List [str]: 
    """
    Very simple word-based chunker.
    For a real system you’d tune sizes based on model context length.
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end = start + max_tokens

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0


    return chunks


def build_prompt(context_chunks: List[str], question: str) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a precise assistant helping with software quality and documentation.
    
    You must:
    - Answer ONLY using the information in the context.
    - If the answer is not in the context, say you don't know.

    Context:
    {context}

    Question:{question}

    Answer:
    """
    return prompt


def get_embedding_function():
    client = get_openai_client()
    model = "text-embedding-3-small"

    def embed(texts:List[str]):
        resp = client.embeddings.create(model=model,input=texts)

        return [item.embedding for item in resp.data]
    
    return embed



