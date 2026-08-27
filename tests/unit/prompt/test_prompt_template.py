from ai_service.prompt.schemas import PromptTemplate


def test_create_sets_defaults() -> None:
    template = PromptTemplate.create(name="rag-qa-system", content="hello {{name}}")

    assert template.version == 1
    assert template.is_active is False
    assert template.variables == []
    assert template.id is None


def test_activate_returns_new_instance_without_mutating_original() -> None:
    template = PromptTemplate.create(name="rag-qa-system", content="c")

    activated = template.activate()

    assert template.is_active is False
    assert activated.is_active is True


def test_render_replaces_variables() -> None:
    template = PromptTemplate.create(
        name="rag-qa-system", content="Hello {{name}}, today is {{date}}."
    )

    result = template.render({"name": "World", "date": "2026-01-01"})

    assert result == "Hello World, today is 2026-01-01."


def test_render_leaves_unknown_placeholders_untouched() -> None:
    template = PromptTemplate.create(name="rag-qa-system", content="{{known}} {{unknown}}")

    result = template.render({"known": "value"})

    assert result == "value {{unknown}}"
