# ---- Build stage ----
FROM python:3.12-slim AS builder

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python dependency manager)
RUN pip install uv

WORKDIR /app

# Copy project files (pyproject.toml defines dependencies)
COPY pyproject.toml ./
# If you use a lock file with uv, uncomment the next line:
# COPY uv.lock ./

# Install runtime dependencies (only, not dev)
RUN uv pip install --system --no-cache-dir

# Copy the app code
COPY app ./app

# ---- Final stage ----
FROM python:3.12-slim

WORKDIR /app

# Copy installed dependences from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY --from=builder /app /app

# Set environment (optional: tune for prod)
ENV PYTHONUNBUFFERED=1

# The command to start your server
CMD ["uvicorn", "app.main:main", "--host", "0.0.0.0", "--port", "8000"]