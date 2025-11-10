# ===============================================
# 🧱 NeuroForge Runner (Sandbox Execution Service)
# ===============================================

FROM python:3.10-slim

WORKDIR /app

# Install compilers and runtimes
RUN apt-get update && apt-get install -y \
    gcc g++ openjdk-21-jdk nodejs npm curl bash docker.io && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt


# Install CPU-only PyTorch first to prevent CUDA downloads
RUN pip install torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu 


RUN pip install --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

EXPOSE 8001
ENV MODE=dev

# Dev-mode live reload
CMD if [ "$MODE" = "dev" ]; then \
      python -m uvicorn app:app --host 0.0.0.0 --port 8001 --reload; \
    else \
      python -m uvicorn app:app --host 0.0.0.0 --port 8001; \
    fi
