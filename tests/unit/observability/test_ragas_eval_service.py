from ai_service.core.config import Settings
from ai_service.observability.schemas import RagasEvalPayload, RagasScores
from ai_service.observability.service import RagasEvalService
from tests.unit.observability.fakes import FakeRagasEvaluationRepository


def _settings() -> Settings:
    return Settings(ragas_llm_eval_enabled=False)


async def test_heuristic_scores_persisted_when_no_llm_eval() -> None:
    repo = FakeRagasEvaluationRepository()
    service = RagasEvalService(repo, _settings())

    await service.evaluate(
        RagasEvalPayload(
            trace_id="trace-1",
            question="what is the capital of korea",
            answer="the capital of korea is seoul",
            contexts=["korea capital seoul information"],
        )
    )

    assert len(repo.saved) == 1
    saved = repo.saved[0]
    assert saved.trace_id == "trace-1"
    assert 0.0 <= saved.faithfulness <= 1.0
    assert 0.0 <= saved.answer_relevancy <= 1.0
    assert saved.context_precision == 0.5


async def test_empty_answer_or_contexts_scores_zero_faithfulness() -> None:
    repo = FakeRagasEvaluationRepository()
    service = RagasEvalService(repo, _settings())

    await service.evaluate(RagasEvalPayload(trace_id="t", question="q", answer="", contexts=[]))

    assert repo.saved[0].faithfulness == 0.0
    assert repo.saved[0].context_precision == 0.0


async def test_llm_eval_used_when_injected() -> None:
    repo = FakeRagasEvaluationRepository()

    async def fake_llm_eval(prompt: str) -> RagasScores:
        return RagasScores(faithfulness=0.9, answer_relevancy=0.8, context_precision=0.7)

    service = RagasEvalService(repo, _settings(), llm_eval=fake_llm_eval)

    await service.evaluate(RagasEvalPayload(trace_id="t", question="q", answer="a", contexts=["c"]))

    saved = repo.saved[0]
    assert saved.faithfulness == 0.9
    assert saved.answer_relevancy == 0.8
    assert saved.context_precision == 0.7


async def test_llm_eval_failure_falls_back_to_heuristic() -> None:
    repo = FakeRagasEvaluationRepository()

    async def failing_llm_eval(prompt: str) -> RagasScores:
        raise ConnectionError("api down")

    service = RagasEvalService(repo, _settings(), llm_eval=failing_llm_eval)

    await service.evaluate(
        RagasEvalPayload(trace_id="t", question="q", answer="a b", contexts=["a b c"])
    )

    assert len(repo.saved) == 1
