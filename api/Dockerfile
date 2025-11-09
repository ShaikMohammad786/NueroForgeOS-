# ===============================================
# 🧠 NeuroForge API Kernel (FastAPI + LangGraph)
# ===============================================

FROM python:3.10-slim

# Working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy the root-level requirements.txt
COPY ../requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt

# Copy API source code (everything in api/)
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# Environment variables
ENV RUNNER_URL=http://runner:8001/run
ENV GEMINI_MODEL=gemini-2.5-flash

# Start FastAPI (since we're building from root context, module path must include package name)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
