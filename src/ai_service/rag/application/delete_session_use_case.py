from typing import Any


class DeleteSessionUseCase:
    def __init__(self, session_repo: Any) -> None:
        self._session_repo = session_repo

    async def execute(self, session_id: str) -> None:
        await self._session_repo.delete_by_id(session_id)


__all__ = ["DeleteSessionUseCase"]
