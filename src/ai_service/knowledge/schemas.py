from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentStatus = Literal["pending", "processed", "failed"]


@dataclass(frozen=True)
class DocumentProps:
    file_name: str
    mime_type: str
    status: DocumentStatus
    chunk_count: int
    id: str | None = None
    created_at: datetime | None = None


class Document:
    def __init__(
        self,
        id: str | None,
        file_name: str,
        mime_type: str,
        status: DocumentStatus,
        chunk_count: int,
        created_at: datetime,
    ) -> None:
        self.id = id
        self.file_name = file_name
        self.mime_type = mime_type
        self.status = status
        self.chunk_count = chunk_count
        self.created_at = created_at

    @classmethod
    def create(cls, file_name: str, mime_type: str) -> "Document":
        return cls(
            id=None,
            file_name=file_name,
            mime_type=mime_type,
            status="pending",
            chunk_count=0,
            created_at=datetime.now(UTC),
        )

    @classmethod
    def restore(cls, props: DocumentProps) -> "Document":
        return cls(
            id=props.id,
            file_name=props.file_name,
            mime_type=props.mime_type,
            status=props.status,
            chunk_count=props.chunk_count,
            created_at=props.created_at or datetime.now(UTC),
        )

    def mark_processed(self, chunk_count: int) -> "Document":
        return Document.restore(
            DocumentProps(
                id=self.id,
                file_name=self.file_name,
                mime_type=self.mime_type,
                status="processed",
                chunk_count=chunk_count,
                created_at=self.created_at,
            )
        )

    def mark_failed(self) -> "Document":
        return Document.restore(
            DocumentProps(
                id=self.id,
                file_name=self.file_name,
                mime_type=self.mime_type,
                status="failed",
                chunk_count=0,
                created_at=self.created_at,
            )
        )


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    document_id: str
    char_count: int
    parent_chunk_id: str | None = None

    @classmethod
    def of(
        cls,
        text: str,
        index: int,
        document_id: str,
        char_count: int | None = None,
        parent_chunk_id: str | None = None,
    ) -> "Chunk":
        return cls(
            text=text,
            index=index,
            document_id=document_id,
            char_count=len(text) if char_count is None else char_count,
            parent_chunk_id=parent_chunk_id,
        )

    def get_text(self) -> str:
        return self.text

    def get_index(self) -> int:
        return self.index

    def get_char_count(self) -> int:
        return self.char_count


@dataclass(frozen=True)
class VectorDocumentMetadata:
    document_id: str
    file_name: str
    chunk_index: int
    char_count: int | None = None
    parent_text: str | None = None
    parent_chunk_id: str | None = None


@dataclass(frozen=True)
class VectorDocument:
    id: str
    text: str
    embedding: list[float]
    metadata: VectorDocumentMetadata


@dataclass(frozen=True)
class SimilaritySearchResult:
    text: str
    score: float
    metadata: VectorDocumentMetadata


@dataclass(frozen=True)
class IngestDocumentCommand:
    file_name: str
    mime_type: str
    content: bytes
    document_id: str | None = None


class DocumentOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    file_name: str = Field(alias="fileName")
    mime_type: str = Field(alias="mimeType")
    status: str
    chunk_count: int = Field(alias="chunkCount")
    created_at: datetime = Field(alias="createdAt")

    @classmethod
    def from_domain(cls, doc: Document) -> "DocumentOut":
        return cls(
            id=doc.id or "",
            fileName=doc.file_name,
            mimeType=doc.mime_type,
            status=doc.status,
            chunkCount=doc.chunk_count,
            createdAt=doc.created_at,
        )


class ChunkOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chunk_index: int = Field(alias="chunkIndex")
    text: str


__all__ = [
    "Chunk",
    "ChunkOut",
    "Document",
    "DocumentOut",
    "DocumentProps",
    "DocumentStatus",
    "IngestDocumentCommand",
    "SimilaritySearchResult",
    "VectorDocument",
    "VectorDocumentMetadata",
]
