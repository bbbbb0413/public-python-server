from datetime import UTC, datetime

from ai_service.feedback.schemas import AnswerFeedback
from ai_service.rag.schemas import ConversationSession, RestoreProps, TurnRecord


class FakeAnswerFeedbackRepository:
    def __init__(self) -> None:
        self.storage: dict[tuple[str, int, str], AnswerFeedback] = {}

    async def upsert(self, feedback: AnswerFeedback) -> AnswerFeedback:
        key = (feedback.session_id, feedback.turn_index, feedback.user_id)
        existing = self.storage.get(key)
        stored = AnswerFeedback(
            session_id=feedback.session_id,
            turn_index=feedback.turn_index,
            user_id=feedback.user_id,
            accuracy=feedback.accuracy,
            helpfulness=feedback.helpfulness,
            comment=feedback.comment,
            # 첫 제출 시각을 지킨다. 실제 저장소의 $setOnInsert 와 같은 규칙이다.
            created_at=existing.created_at if existing else feedback.created_at,
            updated_at=feedback.updated_at,
        )
        self.storage[key] = stored
        return stored

    async def find_one(
        self, session_id: str, turn_index: int, user_id: str
    ) -> AnswerFeedback | None:
        return self.storage.get((session_id, turn_index, user_id))

    async def find_by_session(self, session_id: str, user_id: str) -> list[AnswerFeedback]:
        return sorted(
            (
                f
                for (s, _, u), f in self.storage.items()
                if s == session_id and u == user_id
            ),
            key=lambda f: f.turn_index,
        )


class FakeSessionRepository:
    def __init__(self) -> None:
        self.storage: dict[str, ConversationSession] = {}

    def add(self, session_id: str, user_id: str, roles: list[str]) -> ConversationSession:
        now = datetime.now(UTC)
        turns = [
            TurnRecord(role=role, content=f"{role} 내용", created_at=now)
            for role in roles
        ]
        session = ConversationSession.restore(
            RestoreProps(
                session_id=session_id,
                user_id=user_id,
                title="제목",
                turns=turns,
                created_at=now,
                updated_at=now,
            )
        )
        self.storage[session_id] = session
        return session

    async def find_by_id(self, session_id: str) -> ConversationSession | None:
        return self.storage.get(session_id)
