from typing import Any


class DeleteSessionUseCase:
    """소유자의 세션만 지운다. 지운 것이 없으면 False.

    HTTP 경로는 `SessionService` 를 쓰고 이 유스케이스는 지금 호출되지 않지만,
    소유권 조건 없이 두면 다음에 배선하는 사람이 그대로 구멍을 연다.
    """

    def __init__(self, session_repo: Any) -> None:
        self._session_repo = session_repo

    async def execute(self, session_id: str, user_id: str) -> bool:
        return await self._session_repo.delete_by_id_for_user(  # type: ignore[no-any-return]
            session_id, user_id
        )


__all__ = ["DeleteSessionUseCase"]
