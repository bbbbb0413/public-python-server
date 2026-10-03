from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass(frozen=True)
class AskCommand:
    question: str
    top_k: int = 15
    tenant: str | None = None
    use_hyde: bool = False
    user_id: str | None = None
    conversation_history: list[dict[str, str]] | None = None
    session_id: str | None = None
    on_progress: ProgressCallback | None = None

