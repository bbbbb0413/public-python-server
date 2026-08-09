from dataclasses import dataclass


@dataclass(frozen=True)
class IngestDocumentCommand:
    file_name: str
    mime_type: str
    content: bytes
    document_id: str | None = None
