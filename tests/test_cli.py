from __future__ import annotations

import json
import subprocess
import sys

import pytest

from hairf import answer_question, get_default_framework


@pytest.fixture(autouse=True)
def reset_framework():
    get_default_framework(reset=True)
    yield
    get_default_framework(reset=True)


def test_answer_question_returns_reasoning_result():
    result = answer_question("Summarize adaptive routing strategies.")
    assert result.answer
    assert 0.0 <= result.confidence <= 1.0


def test_cli_invocation_returns_output(tmp_path):
    question = "What is the purpose of the compute-optimal budgeter?"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "hairf",
            "--open-think-pro",
            question,
            "--show-traces",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout
    assert "Answer:" in stdout
    assert "Confidence:" in stdout
    assert question[:20] not in completed.stderr


def test_cli_accepts_json_llm_configuration():
    config = {"model": "demo-llm", "temperature": 0.2}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "hairf",
            "--open-think-pro",
            "Explain the learner.",
            "--llm",
            json.dumps(config),
            "--summary",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "HAIRF Summary" in completed.stdout
