FROM python:3.11

WORKDIR /app

# 1. Install build tools and OpenBLAS for Ryzen math performance
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the optimization flags for your Ryzen 5 5600G
# This ensures that when pip installs from your requirements, it compiles for AVX2
ENV CMAKE_ARGS="-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_OPENBLAS=ON"
ENV FORCE_CMAKE=1

COPY ai/requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# 3. Install your requirements (including llama-cpp-python)
# Because of the CMAKE_ARGS above, llama-cpp-python will be built with AVX2 support
RUN pip install --no-cache-dir -r requirements.txt

# COPY the folder as a package
COPY ai/ /app/ai

ENV PYTHONPATH=/app:$PYTHONPATH
RUN mkdir -p /app/uploads /app/models

EXPOSE 8000

CMD ["uvicorn", "ai.ai_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]