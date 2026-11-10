import base64
import hashlib
import hmac
import json
import os
import threading
import time

class WorkerAuthError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


ISSUER = "anime-backend"
AUDIENCE = "anime-cpu-worker"
CLOCK_SKEW_SECONDS = 60
MAX_REPLAY_ENTRIES = 10000

_REPLAY_LOCK = threading.Lock()
_REPLAY_CACHE = {}

SCOPE_BY_PATH = {
    "/v1/tts/svara": "tts",
    "/v1/tts/supertonic": "tts",
    "/v1/ffmpeg/render-episode": "ffmpeg",
    "/health": "health",
}


def _secret() -> str:
    return os.getenv("WORKER_JOB_TOKEN_SECRET") or os.getenv("WORKER_CALLBACK_SECRET") or ""


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _purge_replay(now: int) -> None:
    for key, expires_at in list(_REPLAY_CACHE.items()):
        if expires_at <= now:
            del _REPLAY_CACHE[key]
    while len(_REPLAY_CACHE) >= MAX_REPLAY_ENTRIES:
        _REPLAY_CACHE.pop(next(iter(_REPLAY_CACHE)))


def _remember_jti(jti: str, expires_at: int) -> None:
    if os.getenv("WORKER_TOKEN_SINGLE_USE", "true").lower() not in {"1", "true", "yes"}:
        return
    now = int(time.time())
    with _REPLAY_LOCK:
        _purge_replay(now)
        if jti in _REPLAY_CACHE:
            raise WorkerAuthError(409, detail="Worker token replay detected.")
        _REPLAY_CACHE[jti] = expires_at


def verify_worker_token(token: str, required_scope: str) -> dict:
    secret = _secret()
    if not secret:
        raise WorkerAuthError(503, detail="Worker authentication is not configured.")
    parts = token.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise WorkerAuthError(401, detail="Malformed worker token.")
    payload_part, signature_part = parts
    expected = hmac.new(secret.encode("utf-8"), payload_part.encode("utf-8"), hashlib.sha256).digest()
    try:
        provided = _b64url_decode(signature_part)
    except Exception as exc:
        raise WorkerAuthError(401, detail="Malformed worker token signature.") from exc
    if not hmac.compare_digest(expected, provided):
        raise WorkerAuthError(401, detail="Worker token signature failed.")
    try:
        claims = json.loads(_b64url_decode(payload_part).decode("utf-8"))
    except Exception as exc:
        raise WorkerAuthError(401, detail="Worker token payload is invalid.") from exc
    if claims.get("iss") != ISSUER or claims.get("aud") != AUDIENCE:
        raise WorkerAuthError(401, detail="Worker token issuer or audience is invalid.")
    if not claims.get("jti") or not claims.get("job_id") or not claims.get("project_id") or not claims.get("user_id"):
        raise WorkerAuthError(401, detail="Worker token is missing required claims.")
    now = int(time.time())
    expires_at = claims.get("exp")
    issued_at = claims.get("iat")
    if not isinstance(expires_at, int) or expires_at <= now:
        raise WorkerAuthError(401, detail="Worker token has expired.")
    if not isinstance(issued_at, int) or issued_at > now + CLOCK_SKEW_SECONDS:
        raise WorkerAuthError(401, detail="Worker token issued-at is invalid.")
    scopes = claims.get("scope")
    if not isinstance(scopes, list) or required_scope not in scopes:
        raise WorkerAuthError(403, detail=f"Worker token is missing scope {required_scope}.")
    _remember_jti(claims["jti"], expires_at)
    return claims


async def worker_auth_middleware(request, call_next):
    from fastapi.responses import JSONResponse
    path = request.url.path.rstrip("/") or "/"
    required_scope = SCOPE_BY_PATH.get(path)
    if required_scope is None:
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    if path == "/health" and os.getenv("WORKER_HEALTH_PUBLIC", "false").lower() in {"1", "true", "yes"}:
        return await call_next(request)
    authorization = request.headers.get("authorization", "")
    if not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"detail": "Bearer worker token required."})
    try:
        request.state.worker_claims = verify_worker_token(authorization[7:].strip(), required_scope)
    except WorkerAuthError as error:
        return JSONResponse(status_code=error.status_code, content={"detail": error.detail})
    return await call_next(request)


def install_worker_auth(app) -> None:
    app.middleware("http")(worker_auth_middleware)
