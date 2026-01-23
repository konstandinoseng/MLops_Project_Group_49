# Drift monitoring job
FROM python:3.11-slim

WORKDIR /app

# Install uv for faster package management
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY data/raw/Sentences_AllAgree.txt data/raw/Sentences_AllAgree.txt

# Install dependencies + extra monitoring deps
RUN uv pip install --system --no-cache . sentence-transformers evidently

# Set default environment variables
ENV GCS_PREFIX="logging"
ENV REFERENCE_DATA_PATH="data/raw/Sentences_AllAgree.txt"
ENV EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Run drift detection
CMD ["python", "-m", "project.monitoring.drift"]
