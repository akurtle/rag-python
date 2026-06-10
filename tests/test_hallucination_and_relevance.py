import math

import pytest

from src.eval_grounding import grounded_answer
from src.rag_chat import retrieve_relevant_chunks
from src.utils import get_embedding_function


# In-scope questions: answers should be grounded in the retrieved context,
# i.e. the model should not be hallucinating extra information.
GROUNDING_CASES = [
    "What types of tests are mentioned?",
    "How do we minimize risk in production?",
    "What architecture does the system follow?",
]

GROUNDING_THRESHOLD = 0.5


@pytest.mark.parametrize("question", GROUNDING_CASES)
def test_answer_is_grounded_in_context(question):
    """
    Hallucination detection: for in-scope questions, most of the words in
    the generated answer should also appear in the retrieved context.
    """
    answer, score = grounded_answer(question, k=4)

    print(f"\nQ: {question}")
    print(f"Answer: {answer}")
    print(f"Grounding score: {score:.2f}")

    assert score >= GROUNDING_THRESHOLD, (
        f"Answer for '{question}' may be hallucinated "
        f"(grounding score {score:.2f} < {GROUNDING_THRESHOLD})"
    )


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


RELEVANCE_CASES = [
    "What types of tests are mentioned?",
    "How do we minimize risk in production?",
    "What architecture does the system follow?",
]

RELEVANCE_THRESHOLD = 0.25


@pytest.mark.parametrize("question", RELEVANCE_CASES)
def test_retrieved_chunks_are_semantically_relevant(question):
    """
    Retrieval relevance validation: the best-matching retrieved chunk
    should be semantically close to the query embedding.
    """
    embed_fn = get_embedding_function()
    query_embedding = embed_fn([question])[0]

    chunks = retrieve_relevant_chunks(question, k=4)
    chunk_embeddings = embed_fn(chunks)

    similarities = [cosine_similarity(query_embedding, ce) for ce in chunk_embeddings]
    top_score = max(similarities)

    print(f"\nQ: {question}")
    print(f"Top chunk similarity: {top_score:.2f}")

    assert top_score >= RELEVANCE_THRESHOLD, (
        f"Top retrieved chunk for '{question}' is not semantically relevant "
        f"(similarity {top_score:.2f} < {RELEVANCE_THRESHOLD})"
    )
