# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.11.23 AS uv
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ORIZZONTE_HOME=/app \
    UV_CACHE_DIR=/app/.cache/uv \
    TEMP=/app/.tmp \
    TMP=/app/.tmp

RUN groupadd --system orizzonte && useradd --system --gid orizzonte --home /app orizzonte
WORKDIR /app
COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY config ./config
RUN uv sync --frozen --no-dev && mkdir -p data/raw data/processed data/manifests models reports logs state .tmp .cache/uv .secrets && chown -R orizzonte:orizzonte /app

USER orizzonte
EXPOSE 8790
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8790/health', timeout=3)"]
ENTRYPOINT ["uv", "run", "--frozen", "orizzonte"]
CMD ["daemon"]
