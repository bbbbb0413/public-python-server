from ai_service.core.config import Settings
from ai_service.llm_gateway.application.llm_routing_service import (
    DEFAULT_CHAIN,
    LlmRoutingService,
)


def _settings(**overrides: object) -> Settings:
    base = {"llm_provider": "ollama", "ollama_model": "m1"}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_falls_back_to_default_chain_when_unconfigured() -> None:
    settings = _settings(llm_provider="unknown-provider", llm_fallback_chain=None)
    service = LlmRoutingService(settings)

    assert service.resolve_chain() == list(DEFAULT_CHAIN)


def test_primary_model_is_placed_first_in_chain() -> None:
    settings = _settings(
        llm_provider="ollama", ollama_model="primary-model", llm_fallback_chain="a,b,primary-model"
    )
    service = LlmRoutingService(settings)

    assert service.resolve_chain() == ["primary-model", "a", "b"]


def test_resolve_chain_with_preferred_model_reorders() -> None:
    settings = _settings(
        llm_provider="ollama", ollama_model="primary-model", llm_fallback_chain="a,b"
    )
    service = LlmRoutingService(settings)

    assert service.resolve_chain(preferred_model="b") == ["b", "primary-model", "a"]


def test_groq_provider_defaults_to_known_model() -> None:
    settings = _settings(llm_provider="groq", groq_model=None, llm_fallback_chain=None)
    service = LlmRoutingService(settings)

    assert service.resolve_chain()[0] == "llama-3.3-70b-versatile"
