from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ai_service.rag.schemas import IterationBudget

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass(frozen=True)
class AgenticAskCommand:
    question: str
    budget: IterationBudget
    top_k: int = 5
    tenant: str | None = None
    confidence_threshold: float = 0.8
    user_id: str | None = None
    conversation_history: list[dict[str, str]] | None = None
    use_hyde: bool = False
    on_progress: ProgressCallback | None = None

