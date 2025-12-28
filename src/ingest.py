import os
import glob

from typing import List, Dict

import chromadb

from chromadb.config import Settings

from pypdf import PdfReader

from .utils import get_embedding_function, get_openai_client, chunk_text

DB_DIR = "./chroma_db"

COLLECTION_NAME = "qa_docs"

DATA_DIR = "./data"

def load_text_files() -> List[Dict]:

    docs=[]

    for path in glob.glob(os.path.join(DATA_DIR,"*.txt")):
        with open(path,"r",encoding= "utf-8") as f:
            text = f.read()

        docs.append({"path":path, "text":text})
    
    return docs

def load_pdf_files()-> List[Dict]:

    docs = []

    for path in glob.glob(os.path.join(DATA_DIR,"*.pdf")):
        reader = PdfReader(path)

        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n".join(pages)

        docs.append({"path":path, "text":text})

    return docs


# def create_embedding_function():
#     client = get_openai_client()
#     model = "text-embedding-3-small"

#     def embed(texts:List[str]):
#         resp = client.embeddings.create(model=model,input=texts)

#         return [item.embedding for item in resp.data]
    
#     return embed

def main():
    
    os.makedirs(DB_DIR, exist_ok=True)

    text_docs = load_text_files()

    pdf_docs = load_pdf_files()

    all_docs = text_docs + pdf_docs


    if not all_docs:
        print("No docs")

        return
    
    chunks = []

    metadatas = []

    ids = []


    for idx, doc in enumerate(all_docs):

        doc_chunks = chunk_text(doc["text"], max_tokens=300, overlap=50)

        print(f"{doc['path']}: split into {len(doc_chunks)} chunks")

        for j, chunk in enumerate(doc_chunks):

            chunks.append(chunk)
            
            metadatas.append({
                "source" : doc["path"],

                "chunk_id": j
            })
            ids.append(f"doc{idx}_chunk{j}")

    
    client = chromadb.PersistentClient(

        path= DB_DIR,

        settings = Settings(allow_reset = True)
    
    )

    try:
        client.delete_collection(COLLECTION_NAME)

    except Exception:
        pass


    collection = client.create_collection(name=COLLECTION_NAME)

    embed_fn = get_embedding_function()

    embeddings = embed_fn(chunks)

    collection.add(
        ids = ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"Indexed {len(chunks)} chunks into collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()