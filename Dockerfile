# ---- Enterprise RAG Platform: container image ----
# Build:  docker build -t rag-platform .
# Run:    docker run -p 8501:8501 -e GROQ_API_KEY=your_key rag-platform

FROM python:3.12-slim

# Don't write .pyc files; print logs straight away; keep the image small.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Run as a normal user, not root (basic container security).
RUN useradd --create-home appuser
WORKDIR /app
RUN chown appuser:appuser /app

# Install dependencies first: Docker caches this layer, so code-only
# changes rebuild in seconds instead of reinstalling everything.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=appuser:appuser . .
USER appuser
RUN mkdir -p /app/data && chown appuser:appuser /app/data

# Download the embedding model at build time, so the container starts fast
# and works without internet access to Hugging Face at runtime.
ARG PRELOAD_MODEL=true
RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
      python -c "from rag.vectorstore import FastEmbedder; FastEmbedder()"; \
    fi

EXPOSE 8501

# Docker marks the container "unhealthy" if Streamlit stops responding.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
