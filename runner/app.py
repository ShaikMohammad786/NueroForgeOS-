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
from threading import BoundedSemaphore
import base64

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator


app = FastAPI(title="NeuroForge Sandbox Runner")

MAX_ARTIFACT_BYTES = int(os.getenv("SANDBOX_MAX_ARTIFACT_BYTES", str(25 * 1024 * 1024)))  # 25 MB default

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
MAX_CONCURRENCY = int(os.getenv("SANDBOX_MAX_CONCURRENCY", "4"))
_RUN_SEMAPHORE = BoundedSemaphore(MAX_CONCURRENCY)
PIP_CACHE_DIR = os.getenv("SANDBOX_PIP_CACHE_DIR")  # host path, e.g. /var/lib/neuroforge/pip-cache


class RunRequest(BaseModel):
    language: str
    code: str
    timeout: int = Field(default=60, gt=0, le=300)
    requirements: Optional[List[str]] = None
    extra_requirements: Optional[List[str]] = None
    network: Optional[str] = Field(default=None, description="Docker network name or 'none'")
    files_b64: Optional[Dict[str, str]] = None  # filename -> base64-encoded content

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
    
    @validator("extra_requirements", each_item=True)
    def _sanitize_extra_requirements(cls, value: str) -> str:
        return value.strip()


def _resolve_image(cfg: SandboxConfig) -> str:
    image = os.getenv(cfg.image_env, cfg.default_image)
    if not image:
        raise RuntimeError(f"No Docker image configured for {cfg.image_env}")
    return image


def _build_create_command(
    cfg: SandboxConfig, container_name: str, network_name: str
) -> List[str]:
    shell_parts: List[str] = ["set -euo pipefail"]
    if cfg.preamble:
        shell_parts.append(cfg.preamble)
    shell_parts.append(cfg.execute)
    shell_cmd = " && ".join(shell_parts)

    cmd: List[str] = [
        "docker",
        "create",
        "--name",
        container_name,
        "--network",
        network_name,
        "--workdir",
        "/workspace",
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

    # Optional shared pip cache to speed up repeated installs
    if cfg.supports_requirements and PIP_CACHE_DIR:
        cmd += ["-v", f"{PIP_CACHE_DIR}:/root/.cache/pip"]

    cmd += [
        _resolve_image(cfg),
        "bash",
        "-lc",
        shell_cmd,
    ]

    return cmd


def _build_start_command(container_name: str) -> List[str]:
    return ["docker", "start", "-a", container_name]


def _docker_cp(src_dir: str, container_name: str, dest_path: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "cp", f"{src_dir}{os.sep}.", f"{container_name}:{dest_path}"],
        capture_output=True,
        text=True,
        check=False,
    )

def _docker_cp_from(container_name: str, src_path: str, dest_dir: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "cp", f"{container_name}:{src_path}", dest_dir],
        capture_output=True,
        text=True,
        check=False,
    )


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
    network_name = req.network if (req.network is not None) else DOCKER_NETWORK

    try:
        _RUN_SEMAPHORE.acquire()

        file_path = os.path.join(temp_dir, cfg.filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(req.code)

        # Optionally materialize provided input files
        if req.files_b64:
            for rel_name, b64 in req.files_b64.items():
                try:
                    data = base64.b64decode(b64)
                    abs_path = os.path.join(temp_dir, rel_name)
                    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                    with open(abs_path, "wb") as outf:
                        outf.write(data)
                except Exception as e:
                    _cleanup_container(container_name)
                    return {
                        "returncode": 1,
                        "stdout": "",
                        "stderr": f"Failed to decode or write input file {rel_name}: {e}",
                    }

        if req.requirements and cfg.supports_requirements:
            requirements_path = os.path.join(temp_dir, "requirements.txt")
            with open(requirements_path, "w", encoding="utf-8") as req_file:
                reqs = list(filter(None, req.requirements))
                if req.extra_requirements:
                    reqs += list(filter(None, req.extra_requirements))
                # de-duplicate while preserving order
                deduped = []
                seen = set()
                for r in reqs:
                    if r not in seen:
                        deduped.append(r)
                        seen.add(r)
                req_file.write("\n".join(deduped))

        # 1) Create container
        create_cmd = _build_create_command(cfg, container_name, network_name)
        create_proc = subprocess.run(create_cmd, capture_output=True, text=True)
        if create_proc.returncode != 0:
            return {
                "returncode": create_proc.returncode,
                "stdout": create_proc.stdout,
                "stderr": create_proc.stderr,
            }

        # 2) Copy workspace into container
        cp_proc = _docker_cp(temp_dir, container_name, "/workspace")
        if cp_proc.returncode != 0:
            _cleanup_container(container_name)
            return {
                "returncode": cp_proc.returncode,
                "stdout": cp_proc.stdout,
                "stderr": cp_proc.stderr or "Failed to docker cp workspace",
            }

        # 3) Start container and stream output
        start_cmd = _build_start_command(container_name)
        result = subprocess.run(start_cmd, capture_output=True, text=True, timeout=req.timeout)

        response: Dict[str, object] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

        # 4) Attempt to collect workspace artifacts into a ZIP (size-limited)
        try:
            temp_out = tempfile.mkdtemp(prefix="nf_out_")
            cp_back = _docker_cp_from(container_name, "/workspace", temp_out)
            if cp_back.returncode == 0:
                # Zip the copied /workspace directory
                workspace_path = os.path.join(temp_out, "workspace")
                # Avoid zipping nothing
                if os.path.exists(workspace_path):
                    zip_base = os.path.join(temp_out, "artifacts")
                    archive_path = shutil.make_archive(zip_base, "zip", workspace_path)
                    try:
                        if os.path.getsize(archive_path) <= MAX_ARTIFACT_BYTES:
                            with open(archive_path, "rb") as fz:
                                b64 = base64.b64encode(fz.read()).decode("utf-8")
                            response["artifacts_zip_b64"] = b64
                        else:
                            response["artifacts_note"] = f"Artifacts exceed size limit ({MAX_ARTIFACT_BYTES} bytes)."
                    finally:
                        # Cleanup temp_out
                        shutil.rmtree(temp_out, ignore_errors=True)
            else:
                response["artifacts_note"] = cp_back.stderr or "Failed to copy workspace from container."
        except Exception as art_exc:
            response["artifacts_note"] = f"Artifact packaging error: {art_exc}"

        return response

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
        try:
            _RUN_SEMAPHORE.release()
        except Exception:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)
