from ai_service.prompt.schemas import PromptTemplate, PromptTemplateProps


class FakePromptTemplateRepository:
    def __init__(self) -> None:
        self.storage: dict[str, PromptTemplate] = {}
        self._next_id = 1

    async def persist(self, template: PromptTemplate) -> PromptTemplate:
        template_id = template.id or str(self._next_id)
        stored = PromptTemplate.restore(_props_with_id(template, template_id))
        self._next_id += 1
        self.storage[template_id] = stored
        return stored

    async def find_by_name_and_version(self, name: str, version: int) -> PromptTemplate | None:
        for t in self.storage.values():
            if t.name.get_value() == name and t.version == version:
                return t
        return None

    async def find_all_by_name(
        self, name: str, user_id: str | None = None
    ) -> list[PromptTemplate]:
        if user_id is not None:
            matches = [
                t
                for t in self.storage.values()
                if t.name.get_value() == name and (t.user_id == user_id or t.user_id is None)
            ]
        else:
            matches = [
                t
                for t in self.storage.values()
                if t.name.get_value() == name and t.user_id is None
            ]
        return sorted(matches, key=lambda t: t.version, reverse=True)

    async def find_active(self, name: str) -> PromptTemplate | None:
        for t in self.storage.values():
            if t.name.get_value() == name and t.user_id is None and t.is_active:
                return t
        return None

    async def find_active_for_user(self, name: str, user_id: str) -> PromptTemplate | None:
        for t in self.storage.values():
            if t.name.get_value() == name and t.user_id == user_id and t.is_active:
                return t
        return None

    async def deactivate_all_by_name(self, name: str) -> None:
        for template_id, t in list(self.storage.items()):
            if t.name.get_value() == name and t.user_id is None and t.is_active:
                self.storage[template_id] = t.deactivate()

    async def deactivate_all_by_name_for_user(self, name: str, user_id: str) -> None:
        for template_id, t in list(self.storage.items()):
            if t.name.get_value() == name and t.user_id == user_id and t.is_active:
                self.storage[template_id] = t.deactivate()

    async def deactivate_active_for_user(self, name: str, user_id: str) -> None:
        await self.deactivate_all_by_name_for_user(name, user_id)

    async def update(self, template: PromptTemplate) -> PromptTemplate:
        template_id = template.id or str(self._next_id)
        self.storage[template_id] = template
        return template


def _props_with_id(template: PromptTemplate, new_id: str) -> PromptTemplateProps:
    return PromptTemplateProps(
        id=new_id,
        name=template.name.get_value(),
        version=template.version,
        content=template.content,
        is_active=template.is_active,
        variables=template.variables,
        user_id=template.user_id,
        created_at=template.created_at,
        updated_at=template.updated_at,
    )
