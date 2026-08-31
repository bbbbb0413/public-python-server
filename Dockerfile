FROM python:3.11-slim AS base

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app
RUN chown -R appuser:appuser /app

FROM base AS build
RUN pip install --no-cache-dir uv

USER appuser
WORKDIR /app

COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN --mount=type=cache,uid=1000,target=/home/appuser/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY --chown=appuser:appuser src ./src
RUN --mount=type=cache,uid=1000,target=/home/appuser/.cache/uv \
    uv sync --locked --no-dev

FROM base
USER appuser
WORKDIR /app

COPY --from=build --chown=appuser:appuser /app /app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 3004
CMD ["uvicorn", "ai_service.main:app", "--host", "0.0.0.0", "--port", "3004"]
