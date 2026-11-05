# Build from the repository root:  docker build -f backend/Dockerfile -t neuroecho .
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
# The extra index provides the much smaller CPU-only PyTorch wheel.
RUN pip install --no-cache-dir -r backend/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY backend backend
COPY ml ml
COPY web web

ENV MODEL_DIR=/app/ml/models
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
