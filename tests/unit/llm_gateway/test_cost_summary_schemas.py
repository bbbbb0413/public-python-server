from datetime import UTC, datetime

from ai_service.llm_gateway.schemas import CostSummaryItem, CostSummaryOut, ModelCostSum


def test_model_cost_sum_default_values() -> None:
    cost_sum = ModelCostSum(model="gpt-4o", total_cost_usd=0.05)
    assert cost_sum.prompt_tokens == 0
    assert cost_sum.completion_tokens == 0
    assert cost_sum.call_count == 0


def test_cost_summary_item_aliases_and_defaults() -> None:
    item = CostSummaryItem(model="gpt-4o", totalCostUsd=0.05)
    assert item.prompt_tokens == 0
    assert item.completion_tokens == 0
    assert item.total_tokens == 0
    assert item.call_count == 0

    dumped = item.model_dump(by_alias=True)
    assert dumped == {
        "model": "gpt-4o",
        "totalCostUsd": 0.05,
        "promptTokens": 0,
        "completionTokens": 0,
        "totalTokens": 0,
        "callCount": 0,
    }


def test_cost_summary_out_of_maps_token_metrics_and_call_count() -> None:
    from_date = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    to_date = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)

    model_sum = ModelCostSum(
        model="gpt-4o",
        total_cost_usd=0.12,
        prompt_tokens=1200,
        completion_tokens=400,
        call_count=5,
    )

    out = CostSummaryOut.of([model_sum], from_date, to_date)
    assert len(out.items) == 1
    item = out.items[0]
    assert item.model == "gpt-4o"
    assert item.total_cost_usd == 0.12
    assert item.prompt_tokens == 1200
    assert item.completion_tokens == 400
    assert item.total_tokens == 1600
    assert item.call_count == 5

    dumped = out.model_dump(by_alias=True)
    assert dumped["items"] == [
        {
            "model": "gpt-4o",
            "totalCostUsd": 0.12,
            "promptTokens": 1200,
            "completionTokens": 400,
            "totalTokens": 1600,
            "callCount": 5,
        }
    ]
