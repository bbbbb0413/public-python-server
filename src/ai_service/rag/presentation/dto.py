from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_service.rag.domain.model.conversation_session import ConversationSession


class TurnOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(alias="createdAt")
    sources: list[dict[str, Any]] | None = None
    confidence: float | None = None
    missing: list[str] | None = None


class SessionOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(alias="sessionId")
    title: str
    updated_at: datetime = Field(alias="updatedAt")

    @classmethod
    def from_domain(cls, session: ConversationSession) -> "SessionOut":
        return cls(
            sessionId=session.get_session_id(),
            title=session.title,
            updatedAt=session.updated_at,
        )


class SessionDetailOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(alias="sessionId")
    title: str
    turns: list[TurnOut]
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    @classmethod
    def from_domain(cls, session: ConversationSession) -> "SessionDetailOut":
        return cls(
            sessionId=session.get_session_id(),
            title=session.title,
            turns=[
                TurnOut(
                    role=t.role,
                    content=t.content,
                    createdAt=t.created_at,
                    sources=t.sources,
                    confidence=t.confidence,
                    missing=t.missing,
                )
                for t in session.turns
            ],
            createdAt=session.created_at,
            updatedAt=session.updated_at,
        )
