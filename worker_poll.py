#!/usr/bin/env python3
"""Pull-mode CPU worker.

The backend enqueues CPU jobs in `worker_jobs`; this process claims one job at a
time over outbound HTTPS, executes the existing FastAPI handler directly, and
posts the result back. No inbound worker URL or tunnel is required.
"""

import json
import os
import time
from typing import Any, Dict, Tuple

import requests
from fastapi import HTTPException

from main import (
    FfmpegEpisodeRequest,
    TtsSynthesisRequest,
    render_episode,
    synthesize_supertonic_endpoint,
    synthesize_svara_endpoint,
)

BACKEND_URL = os.getenv("ANIME_BACKEND_URL", "https://api.bnai.dpdns.org").rstrip("/")
SERVICE_SECRET = (
    os.getenv("WORKER_SERVICE_SECRET")
    or os.getenv("WORKER_JOB_TOKEN_SECRET")
    or ""
).strip()
WORKER_ID = (
    os.getenv("WORKER_ID")
    or (f"github-{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else "local")
).strip()
POLL_INTERVAL_SECONDS = max(1, int(os.getenv("WORKER_POLL_INTERVAL_SECONDS", "5")))
IDLE_EXIT_SECONDS = max(60, int(os.getenv("WORKER_IDLE_EXIT_SECONDS", "1800")))
MAX_LIFETIME_SECONDS = max(300, int(os.getenv("WORKER_MAX_LIFETIME_SECONDS", "21000")))
LEASE_SECONDS = max(60, min(3600, int(os.getenv("WORKER_JOB_LEASE_SECONDS", "1800"))))

DISPATCH = {
    "/v1/tts/svara": (TtsSynthesisRequest, synthesize_svara_endpoint),
    "/v1/tts/supertonic": (TtsSynthesisRequest, synthesize_supertonic_endpoint),
    "/v1/ffmpeg/render-episode": (FfmpegEpisodeRequest, render_episode),
}


def log(message: str) -> None:
    print(f"[cpu-poll] {message}", flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    return str(value)


def session() -> requests.Session:
    client = requests.Session()
    client.headers.update({
        "Authorization": f"Bearer {SERVICE_SECRET}",
        "Content-Type": "application/json",
        "User-Agent": "anime-cpu-pull-worker/1.0",
    })
    return client


def claim(client: requests.Session) -> Dict[str, Any] | None:
    response = client.post(
        f"{BACKEND_URL}/v1/anime/internal/worker/jobs/claim",
        json={"worker_id": WORKER_ID, "lease_seconds": LEASE_SECONDS},
        timeout=30,
    )
    if response.status_code == 204:
        return None
    if response.status_code in (401, 403):
        raise PermissionError(f"claim rejected with HTTP {response.status_code}")
    response.raise_for_status()
    payload = response.json()
    job = payload.get("job") if isinstance(payload, dict) else None
    return job if isinstance(job, dict) else None


def execute(job: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, str | None, int]:
    path = str(job.get("path") or "")
    spec = DISPATCH.get(path)
    if spec is None:
        return None, f"Unsupported worker path: {path}", 400
    model, handler = spec
    payload = job.get("payload")
    if not isinstance(payload, dict):
        return None, "Worker job payload must be an object.", 400
    try:
        request = model(**payload)
        result = handler(request)
        return jsonable(result), None, 200
    except HTTPException as exc:
        return None, str(exc.detail)[:4000], int(exc.status_code)
    except Exception as exc:
        return None, str(exc)[:4000], 500


def complete(
    client: requests.Session,
    job_id: str,
    status: str,
    result: Dict[str, Any] | None,
    error: str | None,
    http_status: int,
) -> None:
    response = client.post(
        f"{BACKEND_URL}/v1/anime/internal/worker/jobs/{job_id}/complete",
        data=json.dumps({
            "worker_id": WORKER_ID,
            "status": status,
            "result": result,
            "error": error,
            "http_status": http_status,
        }, default=str),
        timeout=30,
    )
    if response.status_code in (401, 403):
        raise PermissionError(f"complete rejected with HTTP {response.status_code}")
    response.raise_for_status()


def main() -> int:
    if not SERVICE_SECRET:
        log("WORKER_SERVICE_SECRET/WORKER_JOB_TOKEN_SECRET is missing")
        return 1
    log(f"backend={BACKEND_URL} worker_id={WORKER_ID} lease={LEASE_SECONDS}s")
    client = session()
    started_at = time.time()
    last_job_at = started_at
    auth_failures = 0
    while True:
        now = time.time()
        if now - started_at >= MAX_LIFETIME_SECONDS:
            log("max lifetime reached; exiting")
            return 0
        if now - last_job_at >= IDLE_EXIT_SECONDS:
            log("idle timeout reached; exiting")
            return 0
        try:
            job = claim(client)
            auth_failures = 0
        except PermissionError as exc:
            auth_failures += 1
            log(f"authentication failed ({auth_failures}/3): {exc}")
            if auth_failures >= 3:
                return 1
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        except Exception as exc:
            log(f"claim failed: {str(exc)[:300]}")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        if not job:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        job_id = str(job.get("id") or "")
        path = str(job.get("path") or "")
        last_job_at = time.time()
        log(f"claimed job={job_id} path={path} attempts={job.get('attempts')}")
        result, error, http_status = execute(job)
        status = "completed" if error is None else "failed"
        try:
            complete(client, job_id, status, result, error, http_status)
            log(f"completed job={job_id} status={status} http={http_status}")
        except PermissionError as exc:
            log(f"authentication failed while completing job={job_id}: {exc}")
            return 1
        except Exception as exc:
            log(f"completion post failed job={job_id}: {str(exc)[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
