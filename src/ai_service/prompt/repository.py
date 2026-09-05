from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from ai_service.prompt.schemas import (
    PromptTemplate,
    PromptTemplateProps,
)

COLLECTION_NAME = "prompt_templates"


class PromptTemplateRepository:
    def __init__(self, db: AsyncIOMotorDatabase[dict[str, Any]]) -> None:
        self._collection = db[COLLECTION_NAME]

    async def persist(self, template: PromptTemplate) -> PromptTemplate:
        name_val = (
            template.name.get_value()
            if hasattr(template.name, "get_value")
            else str(template.name)
        )
        record: dict[str, Any] = {
            "name": name_val,
            "version": template.version,
            "content": template.content,
            "isActive": template.is_active,
            "variables": template.variables,
            "createdAt": template.created_at,
            "updatedAt": template.updated_at,
        }
        if template.user_id is not None:
            record["userId"] = template.user_id

        result = await self._collection.insert_one(record)
        return self._to_domain({**record, "_id": result.inserted_id})

    async def find_by_name_and_version(self, name: str, version: int) -> PromptTemplate | None:
        record = await self._collection.find_one({"name": name, "version": version})
        return self._to_domain(record) if record else None

    async def find_all_by_name(
        self, name: str, user_id: str | None = None
    ) -> list[PromptTemplate]:
        query: dict[str, Any]
        if user_id is not None:
            query = {
                "name": name,
                "$or": [{"userId": user_id}, {"userId": {"$exists": False}}],
            }
        else:
            query = {"name": name, "userId": {"$exists": False}}
        cursor = self._collection.find(query).sort("version", -1)
        records = await cursor.to_list(length=None)
        return [self._to_domain(r) for r in records]

    async def find_active(self, name: str) -> PromptTemplate | None:
        record = await self._collection.find_one(
            {"name": name, "userId": {"$exists": False}, "isActive": True}
        )
        return self._to_domain(record) if record else None

    async def find_active_for_user(self, name: str, user_id: str) -> PromptTemplate | None:
        record = await self._collection.find_one(
            {"name": name, "userId": user_id, "isActive": True}
        )
        return self._to_domain(record) if record else None

    async def deactivate_all_by_name(self, name: str) -> None:
        await self._collection.update_many(
            {"name": name, "userId": {"$exists": False}},
            {"$set": {"isActive": False, "updatedAt": datetime.now(UTC)}},
        )

    async def deactivate_all_by_name_for_user(self, name: str, user_id: str) -> None:
        await self._collection.update_many(
            {"name": name, "userId": user_id},
            {"$set": {"isActive": False, "updatedAt": datetime.now(UTC)}},
        )

    async def deactivate_active_for_user(self, name: str, user_id: str) -> None:
        await self._collection.update_many(
            {"name": name, "userId": user_id, "isActive": True},
            {"$set": {"isActive": False, "updatedAt": datetime.now(UTC)}},
        )

    async def update(self, template: PromptTemplate) -> PromptTemplate:
        await self._collection.update_one(
            {"_id": ObjectId(template.id)},
            {"$set": {"isActive": template.is_active, "updatedAt": template.updated_at}},
        )
        return template

    @staticmethod
    def _to_domain(record: dict[str, Any]) -> PromptTemplate:
        return PromptTemplate.restore(
            PromptTemplateProps(
                id=str(record["_id"]) if record.get("_id") is not None else None,
                name=record["name"],
                version=record["version"],
                content=record["content"],
                is_active=record["isActive"],
                variables=record["variables"],
                user_id=record.get("userId"),
                created_at=record["createdAt"],
                updated_at=record["updatedAt"],
            )
        )


# Alias for backward compatibility
PromptTemplateRepositoryImpl = PromptTemplateRepository

__all__ = ["COLLECTION_NAME", "PromptTemplateRepository", "PromptTemplateRepositoryImpl"]
