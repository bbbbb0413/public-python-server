from collections.abc import AsyncIterator

import pytest

from ai_service.rag.application.stream_filter import (
    FILTERED_LINE_PATTERNS,
    filter_token_stream,
    is_filtered_line,
)


def test_filtered_line_patterns_defined() -> None:
    assert len(FILTERED_LINE_PATTERNS) >= 4


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("## Step 1", True),
        ("# Step 2", True),
        ("Step 1: 분석 시작", True),
        ("  step 3: 공백 포함 단계", True),
        ("The final answer is 42", True),
        ("In conclusion, this is correct.", True),
        ("To summarize: all clear", True),
        (r"정답은 \boxed{100} 입니다", True),
        ("일반적인 답변 문장입니다.", False),
        ("Step by step 가이드를 안내합니다.", False),
        ("This is the final countdown.", False),
    ],
)
def test_is_filtered_line(line: str, expected: bool) -> None:
    assert is_filtered_line(line) is expected


@pytest.mark.asyncio
async def test_filter_token_stream_removes_cot_lines_and_collapses_blanks() -> None:
    tokens = [
        "Step 1: 분석 단계\n",
        "\n",
        "\n",
        "\n",
        "문서 내용에 따르면 ",
        "정답입니다.\n",
        "## Step 2\n",
        "중간 설명입니다.\n",
        "The final answer is 정답",
    ]

    async def token_generator() -> AsyncIterator[str]:
        for t in tokens:
            yield t

    result_chunks = [c async for c in filter_token_stream(token_generator())]
    combined = "".join(result_chunks)

    assert "Step 1: 분석 단계" not in combined
    assert "## Step 2" not in combined
    assert "The final answer is 정답" not in combined
    assert "문서 내용에 따르면 정답입니다.\n" in combined
    assert "중간 설명입니다.\n" in combined
    assert "\n\n\n" not in combined
