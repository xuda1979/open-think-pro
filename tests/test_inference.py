import pytest

from hairf.framework import HAIRF
from hairf.inference import (
    LLMConfig,
    MissingCredentialsError,
    ensure_llm_config,
    generate_text,
)
from hairf.types import Query


def test_ensure_llm_config_from_string():
    config = ensure_llm_config("openai/gpt-4o-mini")
    assert isinstance(config, LLMConfig)
    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"


def test_generate_text_with_mock(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        options={"mock_response": "Mock output", "temperature": 0.1},
    )
    response = generate_text("Hello", config)
    assert response.text == "Mock output"
    assert response.provider == "openai"


def test_openai_missing_credentials_mentions_expected_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = LLMConfig(provider="openai", model="gpt-4o-mini")
    with pytest.raises(MissingCredentialsError) as exc:
        generate_text("Hello", config)
    assert "OPENAI_API_KEY" in str(exc.value)


def test_hairf_process_with_llm(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    framework = HAIRF()
    query = Query(text="Design adaptive routing in the framework")
    llm_spec = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "options": {"mock_response": "Reasoning via mock LLM."},
    }
    result = framework.process(query, llm=llm_spec)
    assert any(
        state.metadata.get("provider") == "openai"
        for state in result.states
        if state.metadata
    )
