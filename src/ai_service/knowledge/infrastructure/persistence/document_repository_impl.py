from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from ai_service.knowledge.domain.model.document import Document, DocumentProps

COLLECTION_NAME = "knowledge_documents"


class DocumentRepositoryImpl:
    def __init__(self, db: AsyncIOMotorDatabase[dict[str, Any]]) -> None:
        self._collection = db[COLLECTION_NAME]

    async def persist(self, document: Document) -> Document:
        record: dict[str, Any] = {
            "fileName": document.file_name,
            "mimeType": document.mime_type,
            "status": document.status,
            "chunkCount": document.chunk_count,
            "createdAt": document.created_at,
        }
        result = await self._collection.insert_one(record)
        return self._to_domain({**record, "_id": result.inserted_id})

    async def find_by_id(self, id: str) -> Document | None:
        record = await self._collection.find_one({"_id": ObjectId(id)})
        return self._to_domain(record) if record else None

    async def find_all(self) -> list[Document]:
        cursor = self._collection.find().sort("createdAt", -1)
        records = await cursor.to_list(length=None)
        return [self._to_domain(r) for r in records]

    async def update(self, document: Document) -> Document:
        await self._collection.update_one(
            {"_id": ObjectId(document.id)},
            {"$set": {"status": document.status, "chunkCount": document.chunk_count}},
        )
        return document

    async def remove(self, id: str) -> None:
        await self._collection.delete_one({"_id": ObjectId(id)})

    @staticmethod
    def _to_domain(record: dict[str, Any]) -> Document:
        return Document.restore(
            DocumentProps(
                id=str(record["_id"]) if record.get("_id") is not None else None,
                file_name=record["fileName"],
                mime_type=record["mimeType"],
                status=record["status"],
                chunk_count=record["chunkCount"],
                created_at=record.get("createdAt") or datetime.now(UTC),
            )
        )
