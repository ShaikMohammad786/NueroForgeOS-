# runner/app.py
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator


app = FastAPI(title="NeuroForge Sandbox Runner")


@dataclass(frozen=True)
class SandboxConfig:
    filename: str
    image_env: str
    default_image: str
    execute: str
    preamble: Optional[str] = None
    supports_requirements: bool = False


SANDBOX_CONFIG: Dict[str, SandboxConfig] = {
    "python": SandboxConfig(
        filename="main.py",
        image_env="SANDBOX_IMAGE_PYTHON",
        default_image="python:3.10-slim",
        preamble="if [ -f requirements.txt ] && [ -s requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi",
        execute="python /workspace/main.py",
        supports_requirements=True,
    ),
    "javascript": SandboxConfig(
        filename="main.js",
        image_env="SANDBOX_IMAGE_NODE",
        default_image="node:20-bullseye",
        execute="node /workspace/main.js",
    ),
    "c": SandboxConfig(
        filename="main.c",
        image_env="SANDBOX_IMAGE_C",
        default_image="gcc:13",
        execute="gcc main.c -std=c11 -O2 -o main && ./main",
    ),
    "cpp": SandboxConfig(
        filename="main.cpp",
        image_env="SANDBOX_IMAGE_CPP",
        default_image="gcc:13",
        execute="g++ main.cpp -std=c++17 -O2 -o main && ./main",
    ),
    "java": SandboxConfig(
        filename="Main.java",
        image_env="SANDBOX_IMAGE_JAVA",
        default_image="openjdk:21-slim",
        execute="javac Main.java && java Main",
    ),
}


DOCKER_NETWORK = os.getenv("SANDBOX_DOCKER_NETWORK", "none")
MEMORY_LIMIT = os.getenv("SANDBOX_MEMORY_LIMIT")  # e.g. "256m"
CPU_LIMIT = os.getenv("SANDBOX_CPU_LIMIT")  # e.g. "0.5"
PID_LIMIT = os.getenv("SANDBOX_PIDS_LIMIT", "64")
TMPFS_SIZE = os.getenv("SANDBOX_TMPFS_SIZE")  # e.g. "64m"
EXTRA_FLAGS = shlex.split(os.getenv("SANDBOX_EXTRA_DOCKER_FLAGS", ""))


class RunRequest(BaseModel):
    language: str
    code: str
    timeout: int = Field(default=8, gt=0, le=30)
    requirements: Optional[List[str]] = None

    @validator("language")
    def _normalize_language(cls, value: str) -> str:
        value = value.lower()
        if value not in SANDBOX_CONFIG:
            raise ValueError(f"Unsupported language: {value}")
        return value

    @validator("requirements", each_item=True)
    def _sanitize_requirements(cls, value: str) -> str:
        # Basic guardrail to avoid shell breaking characters
        return value.strip()


def _resolve_image(cfg: SandboxConfig) -> str:
    image = os.getenv(cfg.image_env, cfg.default_image)
    if not image:
        raise RuntimeError(f"No Docker image configured for {cfg.image_env}")
    return image


def _build_docker_command(
    cfg: SandboxConfig, temp_dir: str, container_name: str
) -> List[str]:
    shell_parts: List[str] = ["set -euo pipefail"]
    if cfg.preamble:
        shell_parts.append(cfg.preamble)
    shell_parts.append(cfg.execute)
    shell_cmd = " && ".join(shell_parts)

    cmd: List[str] = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        DOCKER_NETWORK,
    ]

    if MEMORY_LIMIT:
        cmd += ["--memory", MEMORY_LIMIT]
    if CPU_LIMIT:
        cmd += ["--cpus", CPU_LIMIT]
    if PID_LIMIT:
        cmd += ["--pids-limit", PID_LIMIT]
    if TMPFS_SIZE:
        cmd += ["--tmpfs", f"/tmp:rw,size={TMPFS_SIZE}"]

    if EXTRA_FLAGS:
        cmd.extend(EXTRA_FLAGS)

    cmd += [
        "-v",
        f"{temp_dir}:/workspace",
        "--workdir",
        "/workspace",
        _resolve_image(cfg),
        "bash",
        "-lc",
        shell_cmd,
    ]

    return cmd


def _cleanup_container(container_name: str) -> None:
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
    except Exception:
        pass


@app.post("/run")
async def run_code(req: RunRequest):
    cfg = SANDBOX_CONFIG.get(req.language)
    if not cfg:
        raise HTTPException(400, f"Unsupported language: {req.language}")

    temp_dir = tempfile.mkdtemp(prefix="nf_")
    container_name = f"nf_{uuid.uuid4().hex[:12]}"

    try:
        file_path = os.path.join(temp_dir, cfg.filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(req.code)

        if req.requirements and cfg.supports_requirements:
            requirements_path = os.path.join(temp_dir, "requirements.txt")
            with open(requirements_path, "w", encoding="utf-8") as req_file:
                req_file.write("\n".join(filter(None, req.requirements)))

        docker_cmd = _build_docker_command(cfg, temp_dir, container_name)
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=req.timeout,
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        _cleanup_container(container_name)
        return {
            "returncode": 124,
            "stdout": "",
            "stderr": "Execution timed out.",
        }
    except FileNotFoundError as exc:
        # Typically raised when Docker CLI is missing
        _cleanup_container(container_name)
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"Docker unavailable: {exc}",
        }
    except Exception as e:
        _cleanup_container(container_name)
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"Runner error: {e}",
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
