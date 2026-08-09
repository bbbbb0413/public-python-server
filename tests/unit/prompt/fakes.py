from ai_service.prompt.domain.model.prompt_template import PromptTemplate


class FakePromptTemplateRepository:
    def __init__(self) -> None:
        self.storage: dict[tuple[str, int], PromptTemplate] = {}
        self._next_id = 1

    async def persist(self, template: PromptTemplate) -> PromptTemplate:
        stored = PromptTemplate.restore(_props_with_id(template, str(self._next_id)))
        self._next_id += 1
        self.storage[(stored.name.get_value(), stored.version)] = stored
        return stored

    async def find_by_name_and_version(self, name: str, version: int) -> PromptTemplate | None:
        return self.storage.get((name, version))

    async def find_all_by_name(self, name: str) -> list[PromptTemplate]:
        return sorted(
            (t for (n, _), t in self.storage.items() if n == name),
            key=lambda t: t.version,
            reverse=True,
        )

    async def find_active(self, name: str) -> PromptTemplate | None:
        for (n, _), t in self.storage.items():
            if n == name and t.is_active:
                return t
        return None

    async def find_active_for_user(self, name: str, user_id: str) -> PromptTemplate | None:
        for (n, _), t in self.storage.items():
            if n == name and t.user_id == user_id and t.is_active:
                return t
        return None

    async def deactivate_all_by_name(self, name: str) -> None:
        for key, t in list(self.storage.items()):
            if key[0] == name and t.is_active:
                self.storage[key] = t.deactivate()

    async def update(self, template: PromptTemplate) -> PromptTemplate:
        self.storage[(template.name.get_value(), template.version)] = template
        return template


def _props_with_id(template: PromptTemplate, new_id: str):
    from ai_service.prompt.domain.model.prompt_template import PromptTemplateProps

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
