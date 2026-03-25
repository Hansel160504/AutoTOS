FROM python:3.11-slim

WORKDIR /app

# Minimal deps — no cmake, no OpenBLAS, no llama-cpp compilation
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY ai/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY ai/ /app/ai
ENV PYTHONPATH=/app:$PYTHONPATH
RUN mkdir -p /app/uploads

EXPOSE 8000
CMD ["uvicorn", "ai.ai_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]