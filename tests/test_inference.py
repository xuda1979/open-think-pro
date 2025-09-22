import json
from urllib import request

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


def test_gemini_uses_new_environment_variable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "new-key")
    config = LLMConfig(
        provider="gemini",
        model="gemini-2.5-pro",
        options={"mock_response": "Gemini mock response"},
    )
    response = generate_text("Hello Gemini", config)
    assert response.text == "Gemini mock response"
    assert response.provider == "gemini"


def test_gemini_falls_back_to_google_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "legacy-key")

    captured = {}

    class DummyResponse:
        def __enter__(self):
            captured["entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            captured["exited"] = True

        def read(self):
            payload = {
                "candidates": [
                    {"content": {"parts": [{"text": "Legacy Gemini response"}]}},
                ]
            }
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(req, timeout=30):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(request, "urlopen", fake_urlopen)

    config = LLMConfig(provider="gemini", model="gemini_deep_think")
    response = generate_text("Hello", config)

    assert captured["entered"] and captured["exited"]
    assert "legacy-key" in captured["url"]
    assert captured["timeout"] == 30
    assert response.text == "Legacy Gemini response"


def test_gemini_accepts_full_model_resource_paths(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "full-path-key")

    captured = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            payload = {
                "candidates": [
                    {"content": {"parts": [{"text": "Full path response"}]}},
                ]
            }
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(req, timeout=30):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(request, "urlopen", fake_urlopen)

    config = LLMConfig(provider="gemini", model="tunedModels/gemini_deep_think")
    response = generate_text("Hello", config)

    assert "/tunedModels/gemini_deep_think:generateContent" in captured["url"]
    assert "models/" not in captured["url"].split(":generateContent")[0].rsplit("/", 1)[-1]
    assert "full-path-key" in captured["url"]
    assert captured["timeout"] == 30
    assert response.text == "Full path response"


def test_gemini_missing_credentials_mentions_both_env_vars(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config = LLMConfig(provider="gemini", model="gemini-2.5-pro")
    with pytest.raises(MissingCredentialsError) as exc:
        generate_text("Hello", config)
    message = str(exc.value)
    assert "GEMINI_API_KEY" in message
    assert "GOOGLE_API_KEY" in message
