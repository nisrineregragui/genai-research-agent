FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: this layer stays cached as long as requirements.txt is unchanged.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code and corpus only. No secrets: credentials arrive at run time (--env-file).
COPY app ./app
COPY data ./data

# Unprivileged user; the embeddings cache (computed at first startup) must be writable.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/.cache \
    && chown -R appuser /app/.cache
USER appuser

EXPOSE 8000

# The slim image has no curl: Python itself checks /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]