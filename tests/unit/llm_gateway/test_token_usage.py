import pytest

from ai_service.llm_gateway.schemas import TokenUsage


def test_of_creates_valid_token_usage() -> None:
    usage = TokenUsage.of(10, 5)

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total() == 15


def test_negative_prompt_tokens_raises() -> None:
    with pytest.raises(ValueError, match="promptTokens"):
        TokenUsage.of(-1, 0)


def test_negative_completion_tokens_raises() -> None:
    with pytest.raises(ValueError, match="completionTokens"):
        TokenUsage.of(0, -1)
