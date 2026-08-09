from dataclasses import dataclass

from ai_service.shared_kernel.value_object import ValueObject


@dataclass(frozen=True)
class ChunkValue:
    text: str
    index: int
    document_id: str
    char_count: int | None = None
    parent_chunk_id: str | None = None


class Chunk(ValueObject[ChunkValue]):
    def _validate(self, value: ChunkValue) -> None:
        if not value.text or not value.text.strip():
            raise ValueError("청크 텍스트는 비어있을 수 없습니다.")
        if value.index < 0:
            raise ValueError("청크 인덱스는 0 이상이어야 합니다.")

    @classmethod
    def of(
        cls,
        text: str,
        index: int,
        document_id: str,
        char_count: int | None = None,
        parent_chunk_id: str | None = None,
    ) -> "Chunk":
        return cls(ChunkValue(text, index, document_id, char_count, parent_chunk_id))

    def get_text(self) -> str:
        return self.get_value().text

    def get_index(self) -> int:
        return self.get_value().index

    def get_document_id(self) -> str:
        return self.get_value().document_id

    def get_char_count(self) -> int | None:
        return self.get_value().char_count

    def get_parent_chunk_id(self) -> str | None:
        return self.get_value().parent_chunk_id
