"""답변 평가의 도메인 값과 API 계약.

평가는 답변 하나에 사용자 하나당 하나만 존재한다. 다시 제출하면 갱신된다.

답변을 가리키는 방법이 특이하다. 대화 턴에는 고유 식별자가 없고 세션 안의
위치로만 구분되므로, `(session_id, turn_index)` 두 값을 합쳐 답변을 지목한다.
턴은 덧붙이기만 하고 중간에서 지우지 않으므로 위치가 나중에 밀리지 않는다.
"""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# 명세서가 정한 5단계. 숫자로 두면 화면마다 라벨이 달라지므로 값 자체를 고정한다.
RATING_MIN = 1
RATING_MAX = 5

MAX_COMMENT_LENGTH = 1000

Rating = Literal[1, 2, 3, 4, 5]


class AnswerFeedback:
    """저장된 평가 하나."""

    def __init__(
        self,
        session_id: str,
        turn_index: int,
        user_id: str,
        accuracy: int,
        helpfulness: int,
        comment: str | None,
        created_at: datetime,
        updated_at: datetime,
        feedback_id: str | None = None,
    ) -> None:
        self._feedback_id = feedback_id
        self._session_id = session_id
        self._turn_index = turn_index
        self._user_id = user_id
        self._accuracy = accuracy
        self._helpfulness = helpfulness
        self._comment = comment
        self._created_at = created_at
        self._updated_at = updated_at

    @classmethod
    def of(
        cls,
        session_id: str,
        turn_index: int,
        user_id: str,
        accuracy: int,
        helpfulness: int,
        comment: str | None = None,
    ) -> "AnswerFeedback":
        now = datetime.now(UTC)
        return cls(
            session_id=session_id,
            turn_index=turn_index,
            user_id=user_id,
            accuracy=accuracy,
            helpfulness=helpfulness,
            comment=comment,
            created_at=now,
            updated_at=now,
        )

    @property
    def feedback_id(self) -> str | None:
        return self._feedback_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def turn_index(self) -> int:
        return self._turn_index

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def accuracy(self) -> int:
        return self._accuracy

    @property
    def helpfulness(self) -> int:
        return self._helpfulness

    @property
    def comment(self) -> str | None:
        return self._comment

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        return self._updated_at


class SubmitFeedbackIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(alias="sessionId", min_length=1)
    turn_index: int = Field(alias="turnIndex", ge=0)
    accuracy: Rating
    helpfulness: Rating
    comment: str | None = Field(default=None, max_length=MAX_COMMENT_LENGTH)


class FeedbackOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(alias="sessionId")
    turn_index: int = Field(alias="turnIndex")
    accuracy: int
    helpfulness: int
    comment: str | None = None
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    @classmethod
    def from_domain(cls, feedback: AnswerFeedback) -> "FeedbackOut":
        return cls(
            sessionId=feedback.session_id,
            turnIndex=feedback.turn_index,
            accuracy=feedback.accuracy,
            helpfulness=feedback.helpfulness,
            comment=feedback.comment,
            createdAt=feedback.created_at,
            updatedAt=feedback.updated_at,
        )


def to_record(feedback: AnswerFeedback) -> dict[str, Any]:
    return {
        "sessionId": feedback.session_id,
        "turnIndex": feedback.turn_index,
        "userId": feedback.user_id,
        "accuracy": feedback.accuracy,
        "helpfulness": feedback.helpfulness,
        "comment": feedback.comment,
        "createdAt": feedback.created_at,
        "updatedAt": feedback.updated_at,
    }


__all__ = [
    "MAX_COMMENT_LENGTH",
    "RATING_MAX",
    "RATING_MIN",
    "AnswerFeedback",
    "FeedbackOut",
    "Rating",
    "SubmitFeedbackIn",
    "to_record",
]
