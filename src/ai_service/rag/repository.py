from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from ai_service.rag.schemas import (
    ConversationSession,
    RestoreProps,
    TurnRecord,
)

COLLECTION_NAME = "conversation_sessions"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 90


class ConversationSessionRepository:
    def __init__(self, db: AsyncIOMotorDatabase[dict[str, Any]]) -> None:
        self._collection = db[COLLECTION_NAME]

    async def ensure_indexes(self) -> None:
        await self._collection.create_index("sessionId", unique=True, name="session_id_unique_idx")
        await self._collection.create_index(
            [("userId", 1), ("updatedAt", -1)], name="user_id_updated_idx"
        )
        await self._collection.create_index(
            "updatedAt", expireAfterSeconds=SESSION_TTL_SECONDS, name="session_ttl_idx"
        )

    async def find_by_id(self, session_id: str) -> ConversationSession | None:
        record = await self._collection.find_one({"sessionId": session_id})
        return self._to_domain(record) if record else None

    async def find_by_user_id(
        self, user_id: str, page: int, limit: int
    ) -> list[ConversationSession]:
        skip = (page - 1) * limit
        cursor = (
            self._collection.find({"userId": user_id}).sort("updatedAt", -1).skip(skip).limit(limit)
        )
        return [self._to_domain(record) async for record in cursor]

    async def persist(self, session: ConversationSession) -> ConversationSession:
        await self._collection.insert_one(self._to_record(session))
        return session

    async def update(self, session: ConversationSession) -> ConversationSession:
        record = self._to_record(session)
        await self._collection.update_one(
            {"sessionId": session.get_session_id()},
            {"$set": {"turns": record["turns"], "updatedAt": record["updatedAt"]}},
        )
        return session

    async def delete_by_id(self, session_id: str) -> None:
        await self._collection.delete_one({"sessionId": session_id})

    @staticmethod
    def _to_domain(record: dict[str, Any]) -> ConversationSession:
        return ConversationSession.restore(
            RestoreProps(
                session_id=record["sessionId"],
                user_id=record["userId"],
                title=record["title"],
                turns=[
                    TurnRecord(
                        role=t["role"],
                        content=t["content"],
                        created_at=t["createdAt"],
                        sources=t.get("sources"),
                        confidence=t.get("confidence"),
                        missing=t.get("missing"),
                    )
                    for t in record["turns"]
                ],
                created_at=record["createdAt"],
                updated_at=record["updatedAt"],
            )
        )

    @staticmethod
    def _to_record(session: ConversationSession) -> dict[str, Any]:
        turns_data: list[dict[str, Any]] = []
        for t in session.turns:
            turn_dict: dict[str, Any] = {
                "role": t.role,
                "content": t.content,
                "createdAt": t.created_at,
            }
            if t.sources is not None:
                turn_dict["sources"] = t.sources
            if t.confidence is not None:
                turn_dict["confidence"] = t.confidence
            if t.missing is not None:
                turn_dict["missing"] = t.missing
            turns_data.append(turn_dict)

        return {
            "sessionId": session.get_session_id(),
            "userId": session.get_user_id(),
            "title": session.title,
            "turns": turns_data,
            "createdAt": session.created_at,
            "updatedAt": session.updated_at,
        }


# Backward compatibility alias
ConversationSessionRepositoryImpl = ConversationSessionRepository

__all__ = [
    "COLLECTION_NAME",
    "ConversationSessionRepository",
    "ConversationSessionRepositoryImpl",
    "SESSION_TTL_SECONDS",
]
