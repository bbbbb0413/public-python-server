import re
from typing import Any

from bson import ObjectId
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
        """소유자를 가리지 않고 세션을 읽는다.

        **HTTP 경로에서 쓰지 않는다.** 요청한 사람이 누구인지 모르는 자리 —
        Kafka 소비자가 자기가 만든 잡의 세션을 이어 쓰는 경우 — 에만 쓴다.
        사용자 요청을 처리하는 곳은 `find_by_id_for_user` 를 쓴다.
        """
        record = await self._collection.find_one({"sessionId": session_id})
        return self._to_domain(record) if record else None

    async def find_by_id_for_user(
        self, session_id: str, user_id: str
    ) -> ConversationSession | None:
        """소유자의 세션만 읽는다. 남의 것이면 없는 것과 같이 취급한다.

        소유권을 쿼리 조건에 넣는 이유는, 읽어 온 뒤 서비스에서 비교하는 방식이
        호출부가 늘 때마다 빠질 수 있기 때문이다. 조건이 여기 있으면 빠뜨릴
        자리가 없다.
        """
        record = await self._collection.find_one({"sessionId": session_id, "userId": user_id})
        return self._to_domain(record) if record else None

    async def find_by_user_id(
        self, user_id: str, page: int, limit: int, keyword: str | None = None
    ) -> list[ConversationSession]:
        query: dict[str, Any] = {"userId": user_id}
        if keyword and keyword.strip():
            regex_cond = {"$regex": re.escape(keyword.strip()), "$options": "i"}
            query["$or"] = [{"title": regex_cond}, {"turns.content": regex_cond}]
        skip = (page - 1) * limit
        cursor = (
            self._collection.find(query).sort("updatedAt", -1).skip(skip).limit(limit)
        )
        return [self._to_domain(record) async for record in cursor]


    async def persist(self, session: ConversationSession) -> ConversationSession:
        await self._collection.insert_one(self._to_record(session))
        return session

    async def update(self, session: ConversationSession) -> ConversationSession:
        record = self._to_record(session)
        await self._collection.update_one(
            {"sessionId": session.get_session_id()},
            {
                "$set": {
                    "turns": record["turns"],
                    "updatedAt": record["updatedAt"],
                    "isBookmarked": record["isBookmarked"],
                }
            },
        )
        return session

    async def delete_by_id(self, session_id: str) -> None:
        """소유자를 가리지 않고 지운다. HTTP 경로에서 쓰지 않는다."""
        await self._collection.delete_one({"sessionId": session_id})

    async def delete_by_id_for_user(self, session_id: str, user_id: str) -> bool:
        """소유자의 세션만 지운다.

        Returns:
            실제로 지웠는지. 없는 세션과 남의 세션 모두 False 다 — 호출부가
            둘을 같은 응답으로 답해 세션 존재 여부를 흘리지 않게 한다.
        """
        result = await self._collection.delete_one(
            {"sessionId": session_id, "userId": user_id}
        )
        return result.deleted_count > 0

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
                        iterations=t.get("iterations"),
                        is_cancelled=bool(t.get("isCancelled", False)),
                        refined_queries=t.get("refinedQueries"),
                    )
                    for t in record["turns"]
                ],
                created_at=record["createdAt"],
                updated_at=record["updatedAt"],
                is_bookmarked=bool(record.get("isBookmarked", False)),
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
            if t.iterations is not None:
                turn_dict["iterations"] = t.iterations
            if t.is_cancelled:
                turn_dict["isCancelled"] = True
            if t.refined_queries is not None:
                turn_dict["refinedQueries"] = t.refined_queries
            turns_data.append(turn_dict)

        return {
            "sessionId": session.get_session_id(),
            "userId": session.get_user_id(),
            "title": session.title,
            "turns": turns_data,
            "createdAt": session.created_at,
            "updatedAt": session.updated_at,
            "isBookmarked": session.is_bookmarked,
        }


# Backward compatibility alias
ConversationSessionRepositoryImpl = ConversationSessionRepository

BOOKMARK_COLLECTION_NAME = "rag_bookmarks"


class BookmarkRepository:
    def __init__(self, db: AsyncIOMotorDatabase[dict[str, Any]]) -> None:
        self._collection = db[BOOKMARK_COLLECTION_NAME]

    async def ensure_indexes(self) -> None:
        await self._collection.create_index(
            [("userId", 1), ("createdAt", -1)], name="user_id_created_at_idx"
        )

    async def create(self, bookmark_dict: dict[str, Any]) -> str:
        doc = dict(bookmark_dict)
        result = await self._collection.insert_one(doc)
        return str(result.inserted_id)

    async def find_by_user_id(
        self, user_id: str, page: int, limit: int, query: str | None = None
    ) -> list[dict[str, Any]]:
        filter_dict: dict[str, Any] = {"userId": user_id}
        if query and query.strip():
            regex = re.escape(query.strip())
            filter_dict["$or"] = [
                {"title": {"$regex": regex, "$options": "i"}},
                {"content": {"$regex": regex, "$options": "i"}},
                {"note": {"$regex": regex, "$options": "i"}},
            ]
        skip = (page - 1) * limit
        cursor = (
            self._collection.find(filter_dict)
            .sort("createdAt", -1)
            .skip(skip)
            .limit(limit)
        )
        items: list[dict[str, Any]] = []
        async for record in cursor:
            record_copy = dict(record)
            if "_id" in record_copy:
                record_copy["id"] = str(record_copy.pop("_id"))
            items.append(record_copy)
        return items

    async def delete_by_id_for_user(self, bookmark_id: str, user_id: str) -> bool:
        id_conditions: list[Any] = [bookmark_id]
        if ObjectId.is_valid(bookmark_id):
            id_conditions.append(ObjectId(bookmark_id))
        result = await self._collection.delete_one(
            {"_id": {"$in": id_conditions}, "userId": user_id}
        )
        if result.deleted_count > 0:
            return True
        result2 = await self._collection.delete_one(
            {"id": bookmark_id, "userId": user_id}
        )
        return result2.deleted_count > 0


__all__ = [
    "BOOKMARK_COLLECTION_NAME",
    "BookmarkRepository",
    "COLLECTION_NAME",
    "ConversationSessionRepository",
    "ConversationSessionRepositoryImpl",
    "SESSION_TTL_SECONDS",
]
