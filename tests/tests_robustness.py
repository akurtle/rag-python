import pytest

from src.rag_chat import retrieve_relevant_chunks, generate_answer


FUZZ_QUERIES = [
    "",
    "??????",
    "aaaaaaaaaaaaaaaaaaaaa",
    "DROP DATABASE billing;",
    "repeat repeat repeat repeat repeat",
]

@pytest.mark.parametrize("query", FUZZ_QUERIES)
def test_system_handles_fuzzy_inputs_without_crashing(query):
    """
    The system should not throw errors on malformed or adversarial inputs.
    At minimum, it should:
    - return some answer
    - not raise exceptions
    """

    chunks = retrieve_relevant_chunks(query,k=4)

    answer = generate_answer(query,chunks)

      
    print(f"\nQuery: '{query}'")
    print(f"Answer: {answer}")

    assert answer is not None
    assert isinstance(answer, str)
    assert len(answer) >= 0
