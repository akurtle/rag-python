import os

import pytest


from src import ingest as ingest_module

CHROMA_DIR = "../chroma_db"



# when you run any pytest command, this fixture will run first and make sure the DB is there.
@pytest.fixture(scope="session", autouse=True)
def ensure_vector_db():
    """
    Session-wide fixture:
    - If chroma_db is missing or empty, run ingestion.
    - Ensures all tests can query the DB.
    """

    needs_ingest = (
        not os.path.exists(CHROMA_DIR)
        or not os.listdir(CHROMA_DIR)
    )


    if needs_ingest:
        print("\n[setup] No chroma_db found, running ingestion...")
        ingest_module.main()
    else:
        print("\n[setup] Using existing chroma_db")

    yield