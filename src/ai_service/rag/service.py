from datetime import UTC, datetime
from typing import Any

from ai_service.rag.repository import (
    BookmarkRepository,
    ConversationSessionRepository,
)
from ai_service.rag.schemas import (
    BookmarkItemOut,
    ConversationSession,
    CreateBookmarkIn,
)


class SessionNotFoundError(LookupError):
    """세션이 없거나 요청한 사람의 것이 아니다.

    두 경우를 구분하지 않는 것이 의도다. 나누어 답하면 세션 id 하나로
    "그런 세션이 있긴 하다" 를 알아낼 수 있다.
    """


class BookmarkNotFoundError(LookupError):
    """북마크가 없거나 요청한 사람의 것이 아니다."""


class SessionService:
    def __init__(self, repo: ConversationSessionRepository | Any) -> None:
        self._repo = repo

    async def get_sessions(
        self, user_id: str, page: int, limit: int, keyword: str | None = None
    ) -> list[ConversationSession]:
        return await self._repo.find_by_user_id(user_id, page, limit, keyword)


    async def get_session(self, session_id: str, user_id: str) -> ConversationSession:
        session = await self._repo.find_by_id_for_user(session_id, user_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    async def delete_session(self, session_id: str, user_id: str) -> None:
        deleted = await self._repo.delete_by_id_for_user(session_id, user_id)
        if not deleted:
            raise SessionNotFoundError(session_id)

    async def set_bookmark(
        self, session_id: str, user_id: str, bookmarked: bool | None = None
    ) -> ConversationSession:
        session = await self._repo.find_by_id_for_user(session_id, user_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        target_state = (not session.is_bookmarked) if bookmarked is None else bookmarked
        updated = session.with_bookmark(target_state)
        await self._repo.update(updated)
        return updated


class BookmarkService:
    def __init__(
        self,
        bookmark_repo: BookmarkRepository | Any,
        session_repo: ConversationSessionRepository | Any,
    ) -> None:
        self._bookmark_repo = bookmark_repo
        self._session_repo = session_repo

    async def create_bookmark(
        self, user_id: str, dto: CreateBookmarkIn
    ) -> BookmarkItemOut:
        session = await self._session_repo.find_by_id_for_user(dto.session_id, user_id)

        title: str | None = None
        content: str | None = None

        if session is not None:
            title = session.title
            if dto.turn_index is not None:
                if 0 <= dto.turn_index < len(session.turns):
                    target_turn = session.turns[dto.turn_index]
                    content = target_turn.content
        else:
            title = f"세션 {dto.session_id}"

        now = datetime.now(UTC)
        doc: dict[str, Any] = {
            "userId": user_id,
            "sessionId": dto.session_id,
            "turnIndex": dto.turn_index,
            "title": title,
            "content": content,
            "note": dto.note,
            "createdAt": now,
        }
        bookmark_id = await self._bookmark_repo.create(doc)

        return BookmarkItemOut(
            id=bookmark_id,
            sessionId=dto.session_id,
            turnIndex=dto.turn_index,
            title=title,
            content=content,
            note=dto.note,
            createdAt=now,
        )

    async def list_bookmarks(
        self, user_id: str, page: int, limit: int, query: str | None = None
    ) -> list[BookmarkItemOut]:
        records = await self._bookmark_repo.find_by_user_id(
            user_id, page, limit, query
        )
        return [BookmarkItemOut.from_dict(r) for r in records]

    async def delete_bookmark(self, bookmark_id: str, user_id: str) -> None:
        deleted = await self._bookmark_repo.delete_by_id_for_user(
            bookmark_id, user_id
        )
        if not deleted:
            raise BookmarkNotFoundError(bookmark_id)


__all__ = [
    "BookmarkNotFoundError",
    "BookmarkService",
    "SessionNotFoundError",
    "SessionService",
]
