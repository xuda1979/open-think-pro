from hairf.framework import HAIRF
from hairf.types import Query


def test_hairf_process_returns_result():
    framework = HAIRF()
    query = Query(text="Design a step-by-step plan to optimize resource usage", task_type="planning")
    result = framework.process(query)
    assert result.answer
    assert result.confidence > 0
    summary = framework.summary()
    assert "HAIRF Summary" in summary
