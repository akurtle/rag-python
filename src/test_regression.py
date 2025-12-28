import json

from .rag_chat import retrieve_relevant_chunks, generate_answer


EXPECTED = json.load(open("../tests/expected_answers.json"))


TOLERANCE = 0.4



def similarity(a,b)->int:

    aw = set(a.lower().split())
    bw = set(b.lower().split())

    return len(aw & bw) / len(aw | bw)


def test_regression():

    for question, expected in EXPECTED.items():
        
        chunks = retrieve_relevant_chunks(question)

        answer = generate_answer(question,chunks)

        score = similarity(answer,expected)
        
        print(f"Q: {question}")
        print(f"Expected: {expected}")
        print(f"Actual:   {answer}")
        print(f"Score: {score:.2f}")

        if score < TOLERANCE:
            print("Consistency regression detected.\n")
        else:
            print("Consistent.\n")


if __name__ == "__main__":
    test_regression()