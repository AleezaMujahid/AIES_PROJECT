import pytest

from app import create_app
from app.inference import Analyzer


class FakeBackend:
    """Deterministic stand-in for a model so tests never need torch or the Drive folder."""
    kind = "fake"
    name = "Fake model"
    version = "fake-1"
    threshold = 0.66
    max_tokens = 510

    def __init__(self, score=0.9, tokens=100):
        self.value, self.tokens, self.calls = score, tokens, 0

    def score(self, code):
        self.calls += 1
        return self.value, self.tokens


@pytest.fixture
def fake():
    return FakeBackend()


@pytest.fixture
def analyzer(fake):
    return Analyzer(fake)


@pytest.fixture
def client(analyzer):
    app = create_app({"TESTING": True}, analyzer=analyzer)
    return app.test_client()


GOOD = "void f(char *s) { char b[8]; strcpy(b, s); }"
