FROM python:3.11

WORKDIR /app

# Install cmake needed to compile llama-cpp-python from source
RUN apt-get update && apt-get install -y cmake build-essential && rm -rf /var/lib/apt/lists/*

COPY ai_packages/ /packages/
COPY ai/requirements.txt .

# Install pre-downloaded packages offline
RUN pip install --no-cache-dir --no-index --find-links=/packages \
    fastapi uvicorn pydantic starlette python-multipart \
    pymupdf python-docx python-pptx numpy diskcache jinja2

# Install llama-cpp-python from internet (needs cmake to compile)
RUN pip install --no-cache-dir llama-cpp-python==0.3.16

COPY ai/ .

RUN mkdir -p /app/uploads /app/models

EXPOSE 8000

CMD ["uvicorn", "ai_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]