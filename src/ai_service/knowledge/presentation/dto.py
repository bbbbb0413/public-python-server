from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ai_service.knowledge.domain.model.document import Document


class DocumentOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str | None
    file_name: str = Field(alias="fileName")
    mime_type: str = Field(alias="mimeType")
    status: str
    chunk_count: int = Field(alias="chunkCount")
    created_at: datetime = Field(alias="createdAt")

    @classmethod
    def from_domain(cls, document: Document) -> "DocumentOut":
        return cls(
            id=document.id,
            fileName=document.file_name,
            mimeType=document.mime_type,
            status=document.status,
            chunkCount=document.chunk_count,
            createdAt=document.created_at,
        )


class ChunkOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chunk_index: int = Field(alias="chunkIndex")
    text: str
