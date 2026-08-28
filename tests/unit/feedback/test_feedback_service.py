import pytest

from ai_service.feedback.schemas import AnswerFeedback, SubmitFeedbackIn
from ai_service.feedback.service import FeedbackError, FeedbackService
from tests.unit.feedback.fakes import FakeAnswerFeedbackRepository, FakeSessionRepository

OWNER = "user-1"
OTHER = "user-2"
SESSION = "session-1"


def build(roles: list[str] | None = None) -> tuple[FeedbackService, FakeAnswerFeedbackRepository]:
    repo = FakeAnswerFeedbackRepository()
    sessions = FakeSessionRepository()
    sessions.add(SESSION, OWNER, roles if roles is not None else ["user", "assistant"])
    return FeedbackService(repo, sessions), repo


def submit_in(turn_index: int = 1, accuracy: int = 5, helpfulness: int = 4, comment=None):
    return SubmitFeedbackIn(
        sessionId=SESSION,
        turnIndex=turn_index,
        accuracy=accuracy,
        helpfulness=helpfulness,
        comment=comment,
    )


async def test_소유자는_자기_답변에_평가를_남긴다() -> None:
    service, _ = build()

    feedback = await service.submit(submit_in(comment="도움이 됐다"), OWNER)

    assert feedback.accuracy == 5
    assert feedback.helpfulness == 4
    assert feedback.comment == "도움이 됐다"


async def test_같은_답변에_다시_제출하면_갱신된다() -> None:
    service, repo = build()
    await service.submit(submit_in(accuracy=2, helpfulness=2), OWNER)

    updated = await service.submit(submit_in(accuracy=5, helpfulness=5), OWNER)

    assert updated.accuracy == 5
    assert len(repo.storage) == 1


async def test_갱신해도_최초_생성_시각은_지킨다() -> None:
    service, _ = build()
    first = await service.submit(submit_in(accuracy=2, helpfulness=2), OWNER)

    updated = await service.submit(submit_in(accuracy=4, helpfulness=4), OWNER)

    assert updated.created_at == first.created_at


async def test_남의_세션에는_평가를_남길_수_없다() -> None:
    service, repo = build()

    with pytest.raises(FeedbackError) as exc:
        await service.submit(submit_in(), OTHER)

    assert exc.value.status == 404
    assert repo.storage == {}


async def test_없는_세션은_남의_세션과_같은_404로_답한다() -> None:
    service, _ = build()

    with pytest.raises(FeedbackError) as exc:
        await service.submit(
            SubmitFeedbackIn(sessionId="없는-세션", turnIndex=1, accuracy=3, helpfulness=3),
            OWNER,
        )

    assert exc.value.status == 404


async def test_범위를_벗어난_턴은_404() -> None:
    service, _ = build()

    with pytest.raises(FeedbackError) as exc:
        await service.submit(submit_in(turn_index=9), OWNER)

    assert exc.value.status == 404


async def test_질문_턴에는_평가를_남길_수_없다() -> None:
    service, _ = build()

    with pytest.raises(FeedbackError) as exc:
        await service.submit(submit_in(turn_index=0), OWNER)

    assert exc.value.status == 400


async def test_세션_평가_조회는_내_것만_돌려준다() -> None:
    repo = FakeAnswerFeedbackRepository()
    sessions = FakeSessionRepository()
    sessions.add(SESSION, OWNER, ["user", "assistant", "user", "assistant"])
    service = FeedbackService(repo, sessions)
    await service.submit(submit_in(turn_index=1), OWNER)
    await service.submit(submit_in(turn_index=3), OWNER)
    # 다른 사용자의 평가가 같은 세션에 섞여 있어도 새어 나가면 안 된다.
    await repo.upsert(
        AnswerFeedback.of(
            session_id=SESSION, turn_index=3, user_id=OTHER, accuracy=1, helpfulness=1
        )
    )

    items = await service.get_for_session(SESSION, OWNER)

    assert [f.turn_index for f in items] == [1, 3]
    assert {f.user_id for f in items} == {OWNER}


async def test_남의_세션_평가는_조회할_수_없다() -> None:
    service, _ = build()

    with pytest.raises(FeedbackError) as exc:
        await service.get_for_session(SESSION, OTHER)

    assert exc.value.status == 404


async def test_평가가_없으면_빈_목록() -> None:
    service, _ = build()

    assert await service.get_for_session(SESSION, OWNER) == []
