from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from ai_service.feedback.schemas import AnswerFeedback, to_record

COLLECTION_NAME = "answer_feedback"


class AnswerFeedbackRepository:
    def __init__(self, db: AsyncIOMotorDatabase[dict[str, Any]]) -> None:
        self._collection = db[COLLECTION_NAME]

    async def upsert(self, feedback: AnswerFeedback) -> AnswerFeedback:
        """같은 답변에 같은 사용자가 다시 제출하면 갱신한다.

        `createdAt` 은 `$setOnInsert` 로 넘겨 첫 제출 시각을 지킨다. 갱신마다
        생성 시각이 밀리면 "언제부터 이 평가가 있었는지" 를 잃는다.
        """
        record = to_record(feedback)
        created_at = record.pop("createdAt")
        key = {
            "sessionId": feedback.session_id,
            "turnIndex": feedback.turn_index,
            "userId": feedback.user_id,
        }
        await self._collection.update_one(
            key,
            {"$set": record, "$setOnInsert": {"createdAt": created_at}},
            upsert=True,
        )
        stored = await self._collection.find_one(key)
        return self._to_domain(stored) if stored else feedback

    async def find_one(
        self, session_id: str, turn_index: int, user_id: str
    ) -> AnswerFeedback | None:
        record = await self._collection.find_one(
            {"sessionId": session_id, "turnIndex": turn_index, "userId": user_id}
        )
        return self._to_domain(record) if record else None

    async def find_by_session(self, session_id: str, user_id: str) -> list[AnswerFeedback]:
        """한 세션에서 이 사용자가 남긴 평가 전부.

        `userId` 를 조건에 넣는 것이 화면 필터가 아니라 권한 경계다. 빼면 다른
        사용자의 평가가 그대로 나간다.
        """
        cursor = self._collection.find({"sessionId": session_id, "userId": user_id}).sort(
            "turnIndex", 1
        )
        records = await cursor.to_list(length=None)
        return [self._to_domain(r) for r in records]

    @staticmethod
    def _to_domain(record: dict[str, Any]) -> AnswerFeedback:
        return AnswerFeedback(
            feedback_id=str(record["_id"]) if record.get("_id") is not None else None,
            session_id=record["sessionId"],
            turn_index=record["turnIndex"],
            user_id=record["userId"],
            accuracy=record["accuracy"],
            helpfulness=record["helpfulness"],
            comment=record.get("comment"),
            created_at=record["createdAt"],
            updated_at=record["updatedAt"],
        )


__all__ = ["COLLECTION_NAME", "AnswerFeedbackRepository"]
