FROM python:3.12.14-alpine3.24@sha256:b64631e04e4920160c50fbe8d8df828f7f35f06f425cb44aa09bca53e708a35a AS runtime

ARG INSTALL_EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
RUN if [ -n "$INSTALL_EXTRAS" ]; then \
      python -m pip install ".[${INSTALL_EXTRAS}]"; \
    else \
      python -m pip install .; \
    fi

RUN addgroup -S -g 10001 appuser \
    && adduser -S -D -h /home/appuser -u 10001 -G appuser appuser \
    && mkdir -p /app/var \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "-m", "razortrust"]
