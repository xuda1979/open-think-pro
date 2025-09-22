"""Utilities for calling third-party language model APIs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Type
from urllib import error, parse, request


class LLMError(RuntimeError):
    """Base error raised for failures while using an LLM backend."""


class MissingCredentialsError(LLMError):
    """Raised when the expected environment variable is not configured."""

    def __init__(self, provider: str, env_var: str) -> None:
        super().__init__(f"Missing credentials for provider '{provider}': set {env_var}.")
        self.provider = provider
        self.env_var = env_var


class LLMCallError(LLMError):
    """Raised when an API call fails even though credentials were provided."""

    def __init__(self, provider: str, model: str, message: str) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.message = message


def _normalize_provider(value: str) -> str:
    return value.strip().lower()


@dataclass(frozen=True)
class LLMConfig:
    """Configuration describing which model should be used for inference."""

    provider: str
    model: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _normalize_provider(self.provider))
        object.__setattr__(self, "options", dict(self.options))

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the configuration."""

        return {
            "provider": self.provider,
            "model": self.model,
            "options": dict(self.options),
        }


def ensure_llm_config(config: Any) -> Optional[LLMConfig]:
    """Coerce ``config`` into an :class:`LLMConfig` instance."""

    if config is None:
        return None
    if isinstance(config, LLMConfig):
        return config
    if isinstance(config, str):
        if "/" not in config:
            raise ValueError(
                "LLM specification strings must use the form 'provider/model'."
            )
        provider, model = config.split("/", 1)
        return LLMConfig(provider=provider, model=model)
    if isinstance(config, Mapping):
        provider = config.get("provider")
        model = config.get("model")
        if not provider or not model:
            raise ValueError("Mapping based LLM specifications must provide 'provider' and 'model'.")
        options = config.get("options", {})
        if options is None:
            options = {}
        if not isinstance(options, Mapping):
            raise ValueError("'options' must be a mapping if provided.")
        return LLMConfig(provider=str(provider), model=str(model), options=dict(options))
    raise TypeError(f"Unsupported LLM configuration type: {type(config)!r}")


@dataclass
class LLMResponse:
    """Response payload returned from a language model call."""

    text: str
    provider: str
    model: str
    raw: Optional[Dict[str, Any]] = None


class BaseLLMClient:
    """Base helper that performs HTTP requests against an LLM service."""

    provider_name: str = "base"
    api_key_env: str = ""
    base_url_env: str = ""
    default_base_url: str = ""
    requires_bearer_token: bool = True

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        if api_key is None and self.api_key_env:
            api_key = os.getenv(self.api_key_env)
        if base_url is None and self.base_url_env:
            base_url = os.getenv(self.base_url_env)
        if base_url is None:
            base_url = self.default_base_url
        self.api_key = api_key or ""
        self.base_url = base_url.rstrip("/") if base_url else ""

    # Public API ---------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        model: str,
        options: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        options_map: MutableMapping[str, Any] = dict(options or {})
        if "mock_response" in options_map:
            return LLMResponse(
                text=str(options_map["mock_response"]),
                provider=self.provider_name,
                model=model,
                raw={"mock": True},
            )
        if options_map.pop("offline", False):
            return self._offline_response(prompt, model)

        if self.requires_bearer_token and not self.api_key:
            raise MissingCredentialsError(self.provider_name, self.api_key_env)

        try:
            payload = self._perform_request(prompt, model=model, options=options_map)
        except MissingCredentialsError:
            raise
        except error.URLError as exc:  # pragma: no cover - network failure path
            raise LLMCallError(self.provider_name, model, str(exc)) from exc
        except Exception as exc:  # pragma: no cover - network failure path
            raise LLMCallError(self.provider_name, model, str(exc)) from exc

        text = self._extract_text(payload)
        return LLMResponse(text=text, provider=self.provider_name, model=model, raw=payload)

    # Implementation hooks ----------------------------------------------
    def _offline_response(self, prompt: str, model: str) -> LLMResponse:
        snippet = prompt.strip().splitlines()[0] if prompt.strip() else ""
        summary = snippet[:80]
        text = f"[{self.provider_name}:{model}] offline stub for '{summary}'"
        return LLMResponse(text=text, provider=self.provider_name, model=model, raw={"offline": True})

    def _perform_request(
        self,
        prompt: str,
        *,
        model: str,
        options: Mapping[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _extract_text(self, payload: Mapping[str, Any]) -> str:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    provider_name = "openai"
    api_key_env = "OPENAI_API_KEY"
    base_url_env = "OPENAI_API_BASE"
    default_base_url = "https://api.openai.com/v1"

    def _perform_request(
        self,
        prompt: str,
        *,
        model: str,
        options: Mapping[str, Any],
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        system_prompt = options.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(options.get("temperature", 0.7)),
        }
        if "max_tokens" in options:
            payload["max_tokens"] = int(options["max_tokens"])

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers=headers)
        timeout = float(options.get("timeout", 30))
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _extract_text(self, payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    segments = []
                    for item in content:
                        if isinstance(item, Mapping) and "text" in item:
                            segments.append(str(item.get("text", "")))
                    if segments:
                        return "".join(segments)
        if "output" in payload:
            return str(payload["output"])
        return json.dumps(payload)


class GeminiClient(BaseLLMClient):
    provider_name = "gemini"
    api_key_env = "GEMINI_API_KEY"
    base_url_env = "GEMINI_API_BASE"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"
    requires_bearer_token = False
    alternate_api_key_env = "GOOGLE_API_KEY"

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__(api_key=api_key, base_url=base_url)
        if not self.api_key and self.alternate_api_key_env:
            fallback = os.getenv(self.alternate_api_key_env, "")
            if fallback:
                self.api_key = fallback

    @staticmethod
    def _normalise_model_path(model: str) -> str:
        """Return the API path component for ``model``.

        Gemini exposes base models under ``models/{name}`` and tuned models under
        ``tunedModels/{id}``. Vertex AI hosted Gemini models use fully qualified
        resource names (``projects/.../locations/.../models/...``). To keep the
        public API ergonomic we allow callers to pass either the bare model name
        (``gemini-2.0-flash``) or the full resource path. This helper ensures we
        only prefix bare names with ``models/`` and leave explicit paths intact.
        """

        model = model.strip()
        if not model:
            raise ValueError("Model name must be a non-empty string.")
        if model.startswith(("models/", "tunedModels/", "projects/")):
            return model
        return f"models/{model}"

    def _perform_request(
        self,
        prompt: str,
        *,
        model: str,
        options: Mapping[str, Any],
    ) -> Dict[str, Any]:
        api_key = self.api_key or os.getenv(self.api_key_env) or (
            os.getenv(self.alternate_api_key_env, "") if self.alternate_api_key_env else ""
        )
        if not api_key:
            expected = self.api_key_env
            if self.alternate_api_key_env:
                expected = f"{expected} or {self.alternate_api_key_env}"
            raise MissingCredentialsError(self.provider_name, expected)
        self.api_key = api_key
        model_path = self._normalise_model_path(model)
        url = f"{self.base_url}/{model_path}:generateContent"
        query_string = parse.urlencode({"key": api_key})
        url = f"{url}?{query_string}"
        headers = {"Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                    ],
                }
            ]
        }
        if "system_prompt" in options:
            payload.setdefault("systemInstruction", {"parts": []})
            payload["systemInstruction"]["parts"].append({"text": str(options["system_prompt"])})
        if "temperature" in options:
            payload["generationConfig"] = {"temperature": float(options["temperature"])}
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers=headers)
        timeout = float(options.get("timeout", 30))
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _extract_text(self, payload: Mapping[str, Any]) -> str:
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates:
            content = candidates[0].get("content") if isinstance(candidates[0], Mapping) else None
            if isinstance(content, Mapping):
                parts = content.get("parts")
                if isinstance(parts, list) and parts:
                    first = parts[0]
                    if isinstance(first, Mapping) and "text" in first:
                        return str(first.get("text", ""))
        return json.dumps(payload)


class DeepSeekClient(BaseLLMClient):
    provider_name = "deepseek"
    api_key_env = "DEEPSEEK_API_KEY"
    base_url_env = "DEEPSEEK_API_BASE"
    default_base_url = "https://api.deepseek.com/v1"

    def _perform_request(
        self,
        prompt: str,
        *,
        model: str,
        options: Mapping[str, Any],
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(options.get("temperature", 0.7)),
        }
        if "max_tokens" in options:
            payload["max_tokens"] = int(options["max_tokens"])
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers=headers)
        timeout = float(options.get("timeout", 30))
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _extract_text(self, payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    return content
        return json.dumps(payload)


class QwenClient(BaseLLMClient):
    provider_name = "qwen"
    api_key_env = "QWEN_API_KEY"
    base_url_env = "QWEN_API_BASE"
    default_base_url = "https://dashscope.aliyuncs.com/api/v1"

    def _perform_request(
        self,
        prompt: str,
        *,
        model: str,
        options: Mapping[str, Any],
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/services/aigc/text-generation/generation"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        system_prompt = options.get("system_prompt", "You are a helpful assistant.")
        messages = [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": prompt},
        ]
        payload: Dict[str, Any] = {
            "model": model,
            "input": {"messages": messages},
        }
        if "temperature" in options:
            payload["parameters"] = {"temperature": float(options["temperature"])}
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers=headers)
        timeout = float(options.get("timeout", 30))
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _extract_text(self, payload: Mapping[str, Any]) -> str:
        output = payload.get("output")
        if isinstance(output, Mapping):
            if "text" in output:
                return str(output.get("text", ""))
            choices = output.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, Mapping):
                    return str(first.get("message", {}).get("content", ""))
        return json.dumps(payload)


CLIENT_REGISTRY: Dict[str, Type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "gemini": GeminiClient,
    "deepseek": DeepSeekClient,
    "qwen": QwenClient,
}


def _init_client(provider: str, *, options: Mapping[str, Any] | None = None) -> BaseLLMClient:
    provider_normalized = _normalize_provider(provider)
    try:
        client_cls = CLIENT_REGISTRY[provider_normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported LLM provider: {provider}") from exc
    init_kwargs: Dict[str, Any] = {}
    if options:
        if "api_key" in options and options["api_key"]:
            init_kwargs["api_key"] = str(options["api_key"])
        if "base_url" in options and options["base_url"]:
            init_kwargs["base_url"] = str(options["base_url"])
    return client_cls(**init_kwargs)


def generate_text(prompt: str, config: LLMConfig) -> LLMResponse:
    """Generate text from the configured provider."""

    client = _init_client(config.provider, options=config.options)
    return client.generate(prompt, model=config.model, options=config.options)

