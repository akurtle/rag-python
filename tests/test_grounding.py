import pytest

from src.rag_chat import retrieve_relevant_chunks, generate_answer


HALLUCINATION_CASES = [
    "Who is the CEO of the company?",
    "When did the company start?",
    "What programming language is used for billing?"
]


@pytest.mark.parametrize("question",HALLUCINATION_CASES)
def test_model_admits_unknown_for_out_of_scope_questions(question):
    """
    For questions not covered by docs, we expect the model
    to admit it doesn't know, instead of hallucinating.
    """

    chunks = retrieve_relevant_chunks(question,k=4)

    answer = generate_answer(question,chunks).lower()

    
    print(f"\nQ:{question}")

    print(f"\nAnswer:{answer}")

    assert (
        "don't know" in answer
        or "not in the context" in answer
        or "not provided in the context" in answer
    ), "Model appears to be hallucinating instead of admitting unknown."