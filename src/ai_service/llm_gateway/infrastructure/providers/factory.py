from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from ai_service.config.settings import Settings
from ai_service.llm_gateway.domain.model.llm_message import LlmOptions
from ai_service.llm_gateway.domain.port.llm_provider_port import ILlmProvider
from ai_service.llm_gateway.infrastructure.providers.langchain_provider import (
    LangChainLlmProvider,
)

ChatModelFactory = Callable[[LlmOptions | None], BaseChatModel]


def build_llm_provider(settings: Settings) -> ILlmProvider:
    provider = settings.llm_provider

    if provider == "openai":
        return LangChainLlmProvider(_openai_factory(settings))
    if provider == "gemini":
        return LangChainLlmProvider(_gemini_factory(settings))
    if provider == "groq":
        return LangChainLlmProvider(_groq_factory(settings))
    if provider == "ollama":
        return LangChainLlmProvider(_ollama_factory(settings))
    return LangChainLlmProvider(_claude_factory(settings))


def _apply_options(kwargs: dict[str, Any], options: LlmOptions | None) -> dict[str, Any]:
    if options is None:
        return kwargs
    if options.temperature is not None:
        kwargs["temperature"] = options.temperature
    if options.max_tokens is not None:
        kwargs["max_tokens"] = options.max_tokens
    return kwargs


def _openai_factory(settings: Settings) -> ChatModelFactory:
    default_model = settings.openai_model or "gpt-4o-mini"

    def factory(options: LlmOptions | None) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "api_key": settings.openai_api_key,
            "model": (options.model if options and options.model else default_model),
        }
        return ChatOpenAI(**_apply_options(kwargs, options))

    return factory


def _gemini_factory(settings: Settings) -> ChatModelFactory:
    default_model = settings.google_model or "gemini-1.5-flash"

    def factory(options: LlmOptions | None) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs: dict[str, Any] = {
            "google_api_key": settings.google_api_key,
            "model": (options.model if options and options.model else default_model),
        }
        return ChatGoogleGenerativeAI(**_apply_options(kwargs, options))

    return factory


def _groq_factory(settings: Settings) -> ChatModelFactory:
    default_model = settings.groq_model or "llama-3.3-70b-versatile"

    def factory(options: LlmOptions | None) -> BaseChatModel:
        from langchain_groq import ChatGroq

        kwargs: dict[str, Any] = {
            "api_key": settings.groq_api_key,
            "model": (options.model if options and options.model else default_model),
        }
        return ChatGroq(**_apply_options(kwargs, options))

    return factory


def _ollama_factory(settings: Settings) -> ChatModelFactory:
    default_model = settings.ollama_model

    def factory(options: LlmOptions | None) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {
            "base_url": settings.ollama_base_url,
            "model": (options.model if options and options.model else default_model),
        }
        return ChatOllama(**_apply_options(kwargs, options))

    return factory


def _claude_factory(settings: Settings) -> ChatModelFactory:
    default_model = settings.claude_model or "claude-sonnet-4-6"

    def factory(options: LlmOptions | None) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic

        kwargs: dict[str, Any] = {
            "api_key": settings.anthropic_api_key,
            "model": (options.model if options and options.model else default_model),
        }
        return ChatAnthropic(**_apply_options(kwargs, options))

    return factory
