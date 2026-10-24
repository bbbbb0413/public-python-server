import re
from collections.abc import AsyncIterator

FILTERED_LINE_PATTERNS = [
    re.compile(r"^\s*##?\s*Step\s*\d+", re.IGNORECASE),
    re.compile(r"^\s*Step\s*\d+\s*:", re.IGNORECASE),
    re.compile(r"^\s*(The\s+final\s+answer\s+is|In\s+conclusion|To\s+summarize)\b", re.IGNORECASE),
    re.compile(r"\\boxed\{"),
]

# 기존 하위 호환성 유지용 alias
_FILTERED_LINE_PATTERNS = FILTERED_LINE_PATTERNS


def is_filtered_line(line: str) -> bool:
    return any(p.search(line) for p in FILTERED_LINE_PATTERNS)


# 기존 하위 호환성 유지용 alias
_is_filtered_line = is_filtered_line


async def filter_token_stream(source: AsyncIterator[str]) -> AsyncIterator[str]:
    line_buffer = ""
    consecutive_blanks = 0

    async for token in source:
        line_buffer += token
        lines = line_buffer.split("\n")
        line_buffer = lines.pop()

        for line in lines:
            if is_filtered_line(line):
                continue
            if line.strip() == "":
                consecutive_blanks += 1
                if consecutive_blanks <= 1:
                    yield "\n"
            else:
                consecutive_blanks = 0
                yield line + "\n"

    if line_buffer and not is_filtered_line(line_buffer):
        yield line_buffer
