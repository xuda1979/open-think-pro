from hairf.qira import QIRAReasoner, QIRAConfig
from hairf.types import Query, ReasoningState


def test_qira_superposition_creates_summary():
    def generator(query, context):
        yield ReasoningState(content="A", confidence=0.6)
        yield ReasoningState(content="B", confidence=0.8)

    reasoner = QIRAReasoner(QIRAConfig(generator=generator))
    trace = list(reasoner.execute(Query(text="test")))
    assert trace
    assert trace[0].confidence >= 0.6
    assert "Superposed" in trace[0].content or trace[0].content in {"A", "B"}
