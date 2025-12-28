import pytest
from src.rag_chat import retrieve_relevant_chunks

# tiny "ground truth" for relevance
TEST_CASES = [
    {
        "question": "What type of releases minimize risk in production?",
        "expected_keywords": ["canary"],
    },
    {
        "question": "What types of tests are mentioned?",
        "expected_keywords": ["unit", "integration", "contract"],
    },
]




# for defining multiple sets of arguments and fixtures for a single test function or class. 
# This allows the same test logic to run multiple times with different inputs, 
# which reduces code duplication and improves the readability and maintainability of your test suite. 
@pytest.mark.parametrize("case", TEST_CASES)
def test_retrieval_contains_expected_keywords(case):
    q = case["question"]
    expected_keywords = case["expected_keywords"]

    chunks = retrieve_relevant_chunks(q, k=4)
    joined = " ".join(chunks).lower()

    hits = [kw for kw in expected_keywords if kw.lower() in joined]
    coverage = len(hits) / len(expected_keywords)

    print(f"\nQ: {q}")
    print(f"Expected keywords: {expected_keywords}")
    print(f"Hits: {hits}, coverage={coverage:.2f}")

    # Require at least 60% of expected keywords to appear somewhere
    assert coverage >= 0.6, f"Retrieval too weak for question: {q}"
