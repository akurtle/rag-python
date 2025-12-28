"""
Simple retrieval evaluation:
Given known Q/A pairs, check if retrieved chunks contain the answer.
"""

from .rag_chat import retrieve_relevant_chunks

# Ground truth dataset (you can expand this later)
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

def evaluate_retrieval(k=4):
    print("\n Retrieval Evaluation\n")

    scores = []
    for case in TEST_CASES:
        q = case["question"]
        expected = case["expected_keywords"]

        chunks = retrieve_relevant_chunks(q, k)
        joined = " ".join(chunks).lower()

        # scoring: % of expected keywords found
        hits = sum(1 for kw in expected if kw.lower() in joined)
        score = hits / len(expected)

        scores.append(score)

        print(f"Q: {q}")
        print(f"Expected keywords: {expected}")
        print(f"Found in chunks: {[kw for kw in expected if kw in joined]}")
        print(f"Score: {score:.2f}\n")

    print(f"Average retrieval score: {sum(scores)/len(scores):.2f}")


if __name__ == "__main__":
    evaluate_retrieval()
