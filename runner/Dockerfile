# ===============================================
# 🧠 NeuroForge Runner — Multi-language Sandbox
# ===============================================

# Base Image (Python + Debian Slim)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# ---------------------------------------------------
# 1️⃣ Install system dependencies & language toolchains
# ---------------------------------------------------
# - gcc, g++ for C/C++
# - openjdk-21-jdk for Java
# - nodejs, npm for JavaScript
# - curl for healthcheck/debug utilities
# ---------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ openjdk-21-jdk nodejs npm curl bash && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------
# 2️⃣ Ensure pip is up-to-date and stable
# ---------------------------------------------------
RUN pip install --upgrade pip setuptools wheel

# ---------------------------------------------------
# 3️⃣ Copy requirements and install dependencies
# ---------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------
# 4️⃣ Copy the runner source code
# ---------------------------------------------------
COPY . .

# ---------------------------------------------------
# 5️⃣ Verify Uvicorn installation (failsafe)
# ---------------------------------------------------
RUN python -m pip show uvicorn >/dev/null 2>&1 || \
    (echo "⚠️ Uvicorn missing, installing manually..." && pip install uvicorn)

# ---------------------------------------------------
# 6️⃣ Create non-root user for safety
# ---------------------------------------------------
RUN useradd -m runneruser
USER runneruser

# ---------------------------------------------------
# 7️⃣ Expose Runner port
# ---------------------------------------------------
EXPOSE 8001

# ---------------------------------------------------
# 8️⃣ Define entrypoint (FastAPI Runner Server)
# ---------------------------------------------------
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
