import json
from pathlib import Path

from src.rag_chat import retrieve_relevant_chunks, generate_answer

# You can move this to a separate JSON file if you prefer
EXPECTED_ANSWERS = {
    "What types of tests are mentioned?":
        "The system runs unit tests, integration tests, and contract tests.",
    "How do we minimize risk in production?":
        "We minimize risk by using canary releases in production."
}

def jaccard_similarity(a: str, b: str) -> float:
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)

def test_answer_regression():
    """
    Semantic regression test:
    Ensure current answers remain similar enough to expected answers.
    """
    TOLERANCE = 0.4  # can tweak this

    for question, expected in EXPECTED_ANSWERS.items():
        chunks = retrieve_relevant_chunks(question, k=4)
        answer = generate_answer(question, chunks)

        score = jaccard_similarity(answer, expected)

        print(f"\nQ: {question}")
        print(f"Expected: {expected}")
        print(f"Actual:   {answer}")
        print(f"Similarity score: {score:.2f}")

        assert score >= TOLERANCE, (
            f"Semantic regression detected for: {question} "
            f"(similarity {score:.2f} < {TOLERANCE})"
        )
