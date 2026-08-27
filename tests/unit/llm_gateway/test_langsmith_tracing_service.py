from typing import Any

from ai_service.core.config import Settings
from ai_service.llm_gateway.application.langsmith_tracing_service import (
    LangSmithTracingService,
    LlmRunParams,
)


class RecordingLangSmithClient:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []

    def create_run(self, **kwargs: Any) -> None:
        self.created.append(kwargs)

    def update_run(self, run_id: str, **kwargs: Any) -> None:
        self.updated.append({"run_id": run_id, **kwargs})


def _params() -> LlmRunParams:
    return LlmRunParams(
        name="llm-gateway",
        messages=[{"role": "user", "content": "hi"}],
        answer="hello",
        model="model-a",
        completion_tokens=1,
        feature="qa",
        tenant="default",
        start_time=0.0,
        end_time=1.0,
    )


async def test_disabled_by_default_does_nothing() -> None:
    settings = Settings(langsmith_tracing=False, langsmith_api_key="key")
    client = RecordingLangSmithClient()
    service = LangSmithTracingService(settings, client_factory=lambda _: client)

    await service.log_llm_run(_params())

    assert client.created == []


async def test_enabled_records_run() -> None:
    settings = Settings(langsmith_tracing=True, langsmith_api_key="key")
    client = RecordingLangSmithClient()
    service = LangSmithTracingService(settings, client_factory=lambda _: client)

    await service.log_llm_run(_params())

    assert len(client.created) == 1
    assert client.created[0]["name"] == "llm-gateway"
    assert len(client.updated) == 1
    assert client.updated[0]["outputs"]["answer"] == "hello"


async def test_client_failure_is_swallowed() -> None:
    settings = Settings(langsmith_tracing=True, langsmith_api_key="key")

    def factory(_: str) -> RecordingLangSmithClient:
        raise ConnectionError("network down")

    service = LangSmithTracingService(settings, client_factory=factory)

    await service.log_llm_run(_params())
