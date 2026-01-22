FROM python:3.12-slim AS base

RUN pip install uv

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY LICENSE LICENSE
COPY README.md README.md
COPY data/processed/phrasebank_AllAgree.pt data/processed/phrasebank_AllAgree.pt
COPY models/text_model_AllAgree.pt models/text_model_AllAgree.pt

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

EXPOSE 8080
ENTRYPOINT ["sh", "-c", "uv run uvicorn src.project.backend:app --host 0.0.0.0 --port ${PORT:-8080}"]
