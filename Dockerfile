FROM python:3.11-slim AS base

RUN useradd --create-home --shell /bin/bash appuser
RUN pip install --no-cache-dir uv

WORKDIR /app
RUN chown -R appuser:appuser /app

FROM base AS build
USER appuser
WORKDIR /app

COPY --chown=appuser:appuser . .
RUN uv venv && uv pip install -e .

FROM base
USER appuser
WORKDIR /app

COPY --from=build --chown=appuser:appuser /app /app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 3004
CMD ["uvicorn", "ai_service.main:app", "--host", "0.0.0.0", "--port", "3004"]
