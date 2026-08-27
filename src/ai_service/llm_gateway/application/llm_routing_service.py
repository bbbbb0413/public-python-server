import logging

from ai_service.core.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_CHAIN = ["claude-sonnet-4-6"]


class LlmRoutingService:
    def __init__(self, settings: Settings) -> None:
        raw = settings.llm_fallback_chain
        raw_chain = [m.strip() for m in raw.split(",") if m.strip()] if raw else list(DEFAULT_CHAIN)

        primary_model = self._resolve_provider_model(settings)
        if primary_model:
            self._chain = [primary_model, *[m for m in raw_chain if m != primary_model]]
        else:
            self._chain = raw_chain

        logger.info("LLM fallback chain: %s", " → ".join(self._chain))

    @staticmethod
    def _resolve_provider_model(settings: Settings) -> str | None:
        provider = settings.llm_provider
        if provider == "groq":
            return settings.groq_model or "llama-3.3-70b-versatile"
        if provider == "ollama":
            return settings.ollama_model
        if provider == "openai":
            return settings.openai_model
        if provider == "claude":
            return settings.claude_model
        if provider == "gemini":
            return settings.google_model
        return None

    def resolve_chain(self, preferred_model: str | None = None) -> list[str]:
        if not preferred_model:
            return list(self._chain)
        without = [m for m in self._chain if m != preferred_model]
        return [preferred_model, *without]
