import pytest
from pydantic import ValidationError

from ai_service.feedback.schemas import (
    MAX_COMMENT_LENGTH,
    AnswerFeedback,
    FeedbackOut,
    SubmitFeedbackIn,
)


def test_평가값은_1에서_5까지만_받는다() -> None:
    for value in (1, 3, 5):
        dto = SubmitFeedbackIn(sessionId="s", turnIndex=1, accuracy=value, helpfulness=value)
        assert dto.accuracy == value


@pytest.mark.parametrize("value", [0, 6, -1])
def test_범위를_벗어난_평가값은_거부한다(value: int) -> None:
    with pytest.raises(ValidationError):
        SubmitFeedbackIn(sessionId="s", turnIndex=1, accuracy=value, helpfulness=3)


def test_음수_턴_번호는_거부한다() -> None:
    with pytest.raises(ValidationError):
        SubmitFeedbackIn(sessionId="s", turnIndex=-1, accuracy=3, helpfulness=3)


def test_빈_세션_id는_거부한다() -> None:
    with pytest.raises(ValidationError):
        SubmitFeedbackIn(sessionId="", turnIndex=0, accuracy=3, helpfulness=3)


def test_의견은_선택이다() -> None:
    dto = SubmitFeedbackIn(sessionId="s", turnIndex=1, accuracy=3, helpfulness=3)
    assert dto.comment is None


def test_너무_긴_의견은_거부한다() -> None:
    with pytest.raises(ValidationError):
        SubmitFeedbackIn(
            sessionId="s",
            turnIndex=1,
            accuracy=3,
            helpfulness=3,
            comment="가" * (MAX_COMMENT_LENGTH + 1),
        )


def test_응답은_camelCase_로_직렬화한다() -> None:
    feedback = AnswerFeedback.of("s", 1, "u", accuracy=4, helpfulness=5, comment="좋다")

    payload = FeedbackOut.from_domain(feedback).model_dump(by_alias=True)

    assert payload["sessionId"] == "s"
    assert payload["turnIndex"] == 1
    assert payload["accuracy"] == 4
    # 평가자 식별자는 응답에 담지 않는다. 남의 평가와 구분할 필요가 없고, 흘릴 이유도 없다.
    assert "userId" not in payload
