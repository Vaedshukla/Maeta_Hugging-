FROM python:3.11-slim

# CRITICAL: HF Spaces requires UID 1000 user
RUN useradd -m -u 1000 user

WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY --chown=user pyproject.toml ./
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "fastapi>=0.115.0" \
    "pydantic>=2.0.0" \
    "uvicorn>=0.24.0" \
    "gradio>=4.0.0"

# Copy environment code
COPY --chown=user . /app

# Install the package (non-editable, as root for system site-packages)
RUN pip install --no-cache-dir .

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# CRITICAL: Must bind to 0.0.0.0 (not 127.0.0.1)
CMD ["uvicorn", "dataclean_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
