from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from ai_service.shared_kernel.aggregate_root import AggregateRoot

DocumentStatus = Literal["pending", "processed", "failed"]


@dataclass(frozen=True)
class DocumentProps:
    file_name: str
    mime_type: str
    status: DocumentStatus
    chunk_count: int
    id: str | None = None
    created_at: datetime | None = None


class Document(AggregateRoot):
    def __init__(
        self,
        id: str | None,
        file_name: str,
        mime_type: str,
        status: DocumentStatus,
        chunk_count: int,
        created_at: datetime,
    ) -> None:
        super().__init__()
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
