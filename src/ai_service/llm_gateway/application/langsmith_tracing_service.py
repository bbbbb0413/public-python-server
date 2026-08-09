import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ai_service.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmRunParams:
    name: str
    messages: list[dict[str, str]]
    answer: str
    model: str
    completion_tokens: int
    feature: str
    tenant: str
    start_time: float
    end_time: float


class ILangSmithClient(Protocol):
    def create_run(self, **kwargs: Any) -> Any: ...

    def update_run(self, run_id: str, **kwargs: Any) -> Any: ...


class LangSmithTracingService:
    def __init__(
        self,
        settings: Settings,
        client_factory: Callable[[str], ILangSmithClient] | None = None,
    ) -> None:
        self._enabled = settings.langsmith_tracing and bool(settings.langsmith_api_key)
        self._api_key = settings.langsmith_api_key
        self._project_name = settings.langsmith_project
        self._client_factory = client_factory or self._default_client_factory

    @staticmethod
    def _default_client_factory(api_key: str) -> ILangSmithClient:
        from langsmith import Client

        return Client(api_key=api_key)  # type: ignore[return-value]

    async def log_llm_run(self, params: LlmRunParams) -> None:
        if not self._enabled or not self._api_key:
            return

        try:
            client = self._client_factory(self._api_key)
            run_id = str(uuid.uuid4())

            client.create_run(
                id=run_id,
                name=params.name,
                run_type="llm",
                inputs={"messages": params.messages},
                start_time=params.start_time,
                project_name=self._project_name,
            )
            client.update_run(
                run_id,
                outputs={
                    "answer": params.answer,
                    "model": params.model,
                    "completion_tokens": params.completion_tokens,
                },
                end_time=params.end_time,
            )
        except Exception as e:  # noqa: BLE001 - 트레이싱 실패는 무시하고 진행
            logger.warning("LangSmith 기록 실패 (무시): %s", e)
