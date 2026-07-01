# Multi-stage build: install deps in a builder, copy only what's needed into a
# slim runtime. Keeps the final image small and free of build tooling.

# ---- builder: resolve and install dependencies ----
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
# --prefix puts the installed packages in a relocatable tree we copy below.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- runtime: slim image, non-root, serves the API ----
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
# Run as an unprivileged user, not root.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Bring in the pre-installed dependencies, then the application source.
COPY --from=builder /install /usr/local
COPY finplan ./finplan
COPY app.py ./app.py
COPY scripts ./scripts

USER appuser
EXPOSE 8000

# Liveness check hits the no-network /healthz route (stdlib only, no curl needed).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "finplan.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
