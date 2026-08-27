import pytest

from ai_service.prompt.schemas import PromptName


def test_valid_name() -> None:
    name = PromptName.of("rag-qa-system")
    assert name.get_value() == "rag-qa-system"


def test_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="비어있을 수 없습니다"):
        PromptName.of("")


@pytest.mark.parametrize("value", ["Rag-QA", "rag_qa", "rag qa", "rag.qa"])
def test_invalid_characters_raise(value: str) -> None:
    with pytest.raises(ValueError, match="소문자, 숫자, 하이픈"):
        PromptName.of(value)
