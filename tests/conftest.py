import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CORPUS = ["Q: What is the chemical symbol for gold?\nA: Au\n\n", "Q: In what country is Osaka?\nA: Japan",
          "What is the occupation of Rahul Sharma? Answer briefly. If you are not sure, say I don't know.",
          "cricketer politician singer India Brazil Paris London"]


@pytest.fixture(scope="session")
def tok():
    from abst.tiny import tiny_tokenizer

    return tiny_tokenizer(CORPUS)


@pytest.fixture(scope="session")
def model(tok):
    from abst.tiny import tiny_model

    return tiny_model(tok)
