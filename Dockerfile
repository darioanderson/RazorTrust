ARG PYTHON_BASE_IMAGE=python:3.12.14-alpine3.24@sha256:b64631e04e4920160c50fbe8d8df828f7f35f06f425cb44aa09bca53e708a35a
FROM ${PYTHON_BASE_IMAGE} AS runtime

ARG INSTALL_EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Alpine 3.24 has not yet published the util-linux fix in its stable index.
# Pull only the ABI-compatible libuuid patch from edge and pin its fixed version;
# Debian-based ML builds skip this branch.
RUN if [ -f /etc/alpine-release ]; then \
      apk add --no-cache \
        --repository https://dl-cdn.alpinelinux.org/alpine/edge/main \
        "libuuid=2.42.3-r0"; \
    fi

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
RUN if [ -f /etc/alpine-release ] && echo ",$INSTALL_EXTRAS," | grep -q ',ml,'; then \
      apk add --no-cache --virtual .ml-build-deps build-base; \
    fi \
    && if [ -n "$INSTALL_EXTRAS" ]; then \
      python -m pip install ".[${INSTALL_EXTRAS}]"; \
    else \
      python -m pip install .; \
    fi \
    && python -m pip uninstall --yes pip setuptools \
    && if command -v apk >/dev/null && apk info --quiet .ml-build-deps; then \
      apk del .ml-build-deps; \
    fi

RUN if [ -f /etc/alpine-release ]; then \
      addgroup -S -g 10001 appuser \
      && adduser -S -D -h /home/appuser -u 10001 -G appuser appuser; \
    else \
      groupadd --gid 10001 appuser \
      && useradd --uid 10001 --gid appuser --home-dir /home/appuser \
        --create-home --shell /usr/sbin/nologin appuser; \
    fi \
    && mkdir -p /app/var \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "-m", "razortrust"]
