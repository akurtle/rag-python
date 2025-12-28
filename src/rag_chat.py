import chromadb

from chromadb.config import Settings

from typing import List 

from .utils import get_openai_client, build_prompt,get_embedding_function


DB_DIR = "./chroma_db"
COLLECTION_NAME = "qa_docs"

def retrieve_relevant_chunks(query: str, k: int =4 ) -> List[str]:
    client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(allow_reset=False)
    )
    collection= client.get_collection(name=COLLECTION_NAME)

    embedfn = get_embedding_function()

    query_embedding = embedfn([query])[0]

    results = collection.query(

        query_embeddings=[query_embedding],

        n_results = k,
    )

    documents = results["documents"][0]

    return documents


def generate_answer(question:str, context_chunks: List[str]) -> str:

    client = get_openai_client()

    prompt = build_prompt(context_chunks,question)


    response = client.chat.completions.create(
        model="gpt-5-nano",  # or gpt-4o-mini / gpt-3.5-turbo depending on your access
        messages=[
            {"role": "system", "content": "You are a concise assistant for software quality and documentation."},
            {"role": "user", "content": prompt},
        ],
        
    )


    return response.choices[0].message.content.strip()



def chat_loop():
    print("RAG QA Assistant (type 'exit' to quit)")
    while True:
        q = input("\nQuestion: ")
        if q.lower() in {"exit", "quit"}:
            break

        print("Retrieving relevant context...")
        context_chunks = retrieve_relevant_chunks(q, k=4)

        print("\n--- Retrieved Context (for debugging / QE) ---")
        for i, c in enumerate(context_chunks):
            print(f"[Chunk {i+1}]\n{c[:300]}...\n")

        answer = generate_answer(q, context_chunks)
        print("\n Answer:")
        print(answer)

if __name__ == "__main__":
    chat_loop()


    