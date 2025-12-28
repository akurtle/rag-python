"""
Check if answer is grounded in retrieved chunks by verifying
the answer only uses tokens that exist in the context.
This is a simple heuristic, but useful.
"""

from .rag_chat import retrieve_relevant_chunks, generate_answer

def grounded_answer(question, k=4):
    chunks = retrieve_relevant_chunks(question, k)
    answer = generate_answer(question, chunks)

    context_words = set(" ".join(chunks).lower().split())
    answer_words = answer.lower().split()

    # % of answer tokens found in context
    grounding_score = sum(1 for w in answer_words if w in context_words) / len(answer_words)

    return answer, grounding_score

if __name__ == "__main__":
    q = "What type of tests do we run?"
    answer, score = grounded_answer(q)
    print(f"Q: {q}")
    print(f"Answer: {answer}")
    print(f"Grounding score: {score:.2f}")
