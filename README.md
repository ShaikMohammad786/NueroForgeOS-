## Sandbox Runner (Ephemeral Containers) - Operations

- Each execution runs in an isolated container created on demand.
- Python dependencies are auto-detected from imports and installed per-run.
- Network is enabled per request (default `bridge`) to allow `pip install` and downloads.
- Concurrency is bounded; tune with `SANDBOX_MAX_CONCURRENCY`.
- Optional pip cache speeds up repeated installs.

### Enable pip cache (recommended)
1. On the Docker host (not inside containers), create a cache directory:

```bash
sudo mkdir -p /var/lib/neuroforge/pip-cache
sudo chmod 777 /var/lib/neuroforge/pip-cache
```

2. Ensure `docker-compose.yml` has:

```yaml
environment:
  - SANDBOX_PIP_CACHE_DIR=/var/lib/neuroforge/pip-cache
```

3. Rebuild and restart runner:

```bash
docker compose build runner
docker compose up -d runner
```

### Networking
- API defaults to `SANDBOX_DEFAULT_NETWORK=bridge` so downloads work.
- You may set `SANDBOX_DOCKER_NETWORK=none` at the runner level for stricter defaults and override per request.

### RAG and Tool Reuse
- Successful tools are stored with a success bias and preferred on future queries.
- Error traces are stored to reduce repeats and improve generations.


