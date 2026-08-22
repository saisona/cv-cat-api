FROM python:3.11-slim AS builder

ENV POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

# Only copy lockfiles to maximize cache hits
COPY pyproject.toml poetry.lock ./

RUN poetry install --only main --no-root


FROM python:3.11-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    TORCH_HOME="/app/.cache/torch"

# Install minimal runtime shared libs (e.g., OpenMP runtime for PyTorch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create application user and directory structure
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/.cache/torch && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy virtual environment
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Switch to non-root user early
USER appuser

# Pre-download and bake model weights into image (Cached independently of application code)
RUN python -c "from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights; mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)"

# Copy application source code LAST (Changes here won't re-trigger dependency install or weight download)
COPY --chown=appuser:appuser main.py batcher.py model.py /app/

# Start server using the PORT env var for cloud orchestration compatibility
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info", "--access-log"]
