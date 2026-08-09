import pytest

from ai_service.llm_gateway.domain.vo.model_route import ModelRoute


def test_of_creates_valid_model_route() -> None:
    route = ModelRoute.of("claude-sonnet-4-6")

    assert route.get_value() == "claude-sonnet-4-6"


def test_empty_model_raises() -> None:
    with pytest.raises(ValueError, match="모델명"):
        ModelRoute.of("")


def test_blank_model_raises() -> None:
    with pytest.raises(ValueError, match="모델명"):
        ModelRoute.of("   ")
