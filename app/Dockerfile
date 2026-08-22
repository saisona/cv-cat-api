FROM python:3.11-slim AS builder

ENV POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl &&\
    rm -rf /var/lib/apt/lists/* && \
    curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

COPY pyproject.toml .
COPY poetry.lock .

RUN poetry install --only main --no-root

FROM python:3.11-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy the built virtual environment from builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Switch to non-root
USER appuser

# Copy application source code
COPY --chown=appuser:appuser main.py /app
COPY --chown=appuser:appuser model.py /app

# Warm-up / Pre-download weights at build time (runs inside non-root home directory)
RUN python -c "from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights; mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)"

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--loop", "uvloop"]
