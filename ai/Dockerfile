FROM python:3.11

WORKDIR /app

# build deps
RUN apt-get update && apt-get install -y cmake build-essential && rm -rf /var/lib/apt/lists/*

COPY ai_packages/ /packages/
COPY ai/requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# install pre-downloaded packages
RUN pip install --no-cache-dir --no-index --find-links=/packages \
    fastapi uvicorn pydantic starlette python-multipart \
    pymupdf python-docx python-pptx numpy diskcache jinja2

# install llama-cpp-python as before (or your chosen method)
RUN pip install --no-cache-dir llama-cpp-python==0.3.16

# COPY the folder as a package so "import ai.*" works
COPY ai/ /app/ai

# make sure /app is on PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

RUN mkdir -p /app/uploads /app/models

EXPOSE 8000

# start using the package module path
CMD ["uvicorn", "ai.ai_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]