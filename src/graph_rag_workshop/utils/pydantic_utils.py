"""Utility functions for working with Pydantic AI models and providers."""

import os
from pathlib import Path

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from graph_rag_workshop.settings import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_VLLM_BASE_URL,
    DEFAULT_VLLM_MODEL,
    LLM_PROVIDER,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VLLM_API_KEY_FILE = PROJECT_ROOT / ".vllm_api_key"
NO_AUTH_API_KEY = "not-required"


def get_vllm_api_key(api_key: str | None = None) -> str:
    """Read the HEIA vLLM API key from an argument, environment, or local file."""
    resolved_api_key = api_key or os.getenv("VLLM_API_KEY")
    if not resolved_api_key and VLLM_API_KEY_FILE.is_file():
        resolved_api_key = VLLM_API_KEY_FILE.read_text(encoding="utf-8").strip()

    if not resolved_api_key:
        raise ValueError(
            "The HEIA vLLM endpoint requires an API key. Create '.vllm_api_key' "
            "in the project root and paste the key provided by the instructors."
        )

    return resolved_api_key


def get_llm_settings(
    provider: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str, str]:
    """Resolve the selected Ollama or HEIA vLLM connection settings."""
    selected_provider = (provider or os.getenv("LLM_PROVIDER", LLM_PROVIDER)).lower()

    if selected_provider == "ollama":
        return (
            model_name or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            api_key or NO_AUTH_API_KEY,
        )
    if selected_provider == "vllm":
        return (
            model_name or os.getenv("VLLM_MODEL", DEFAULT_VLLM_MODEL),
            base_url or os.getenv("VLLM_BASE_URL", DEFAULT_VLLM_BASE_URL),
            get_vllm_api_key(api_key),
        )

    raise ValueError("LLM_PROVIDER must be either 'ollama' or 'vllm'.")


def get_llm_model(
    provider: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAIChatModel:
    """Create a Pydantic AI model for local Ollama or HEIA vLLM.

    Args:
        provider: ``ollama`` (default) or ``vllm``.
        model_name: Optional override for the selected provider's model.
        base_url: Optional override for the selected provider's API URL.
        api_key: Optional API key override.

    Returns:
        An OpenAIChatModel configured for the selected OpenAI-compatible API.
    """
    resolved_model, resolved_base_url, resolved_api_key = get_llm_settings(
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
    )

    return OpenAIChatModel(
        model_name=resolved_model,
        provider=OpenAIProvider(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
        ),
    )
