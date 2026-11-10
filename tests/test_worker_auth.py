import base64
import hashlib
import hmac
import json
import time

import pytest
import worker_auth


def _token(secret: str, *, scope=("tts",), jti="jti-1", exp_offset=300, iss="anime-backend", aud="anime-cpu-worker"):
    now = int(time.time())
    claims = {
        "iss": iss,
        "aud": aud,
        "sub": "job-1",
        "job_id": "job-1",
        "project_id": "project-1",
        "tenant_id": "tenant-1",
        "user_id": "user-1",
        "scope": list(scope),
        "iat": now,
        "exp": now + exp_offset,
        "jti": jti,
    }
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode("utf-8")).rstrip(b"=").decode("ascii")
    signature = base64.urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()
    ).rstrip(b"=").decode("ascii")
    return f"{payload}.{signature}"


def test_worker_token_accepts_matching_scope(monkeypatch):
    monkeypatch.setenv("WORKER_JOB_TOKEN_SECRET", "test-secret")
    claims = worker_auth.verify_worker_token(_token("test-secret", jti="ok-1"), "tts")
    assert claims["job_id"] == "job-1"


def test_worker_token_rejects_wrong_scope(monkeypatch):
    monkeypatch.setenv("WORKER_JOB_TOKEN_SECRET", "test-secret")
    with pytest.raises(worker_auth.WorkerAuthError) as error:
        worker_auth.verify_worker_token(_token("test-secret", jti="ok-2"), "ffmpeg")
    assert error.value.status_code == 403


def test_worker_token_rejects_tampering(monkeypatch):
    monkeypatch.setenv("WORKER_JOB_TOKEN_SECRET", "test-secret")
    payload, signature = _token("test-secret", jti="ok-3").split(".")
    with pytest.raises(worker_auth.WorkerAuthError) as error:
        worker_auth.verify_worker_token(f"{payload[:-1]}A.{signature}", "tts")
    assert error.value.status_code == 401


def test_worker_token_rejects_expiry(monkeypatch):
    monkeypatch.setenv("WORKER_JOB_TOKEN_SECRET", "test-secret")
    with pytest.raises(worker_auth.WorkerAuthError) as error:
        worker_auth.verify_worker_token(_token("test-secret", jti="ok-4", exp_offset=-1), "tts")
    assert error.value.status_code == 401


def test_worker_token_rejects_replay(monkeypatch):
    monkeypatch.setenv("WORKER_JOB_TOKEN_SECRET", "test-secret")
    monkeypatch.setenv("WORKER_TOKEN_SINGLE_USE", "true")
    token = _token("test-secret", jti="replay-1")
    worker_auth.verify_worker_token(token, "tts")
    with pytest.raises(worker_auth.WorkerAuthError) as error:
        worker_auth.verify_worker_token(token, "tts")
    assert error.value.status_code == 409


def test_worker_token_rejects_wrong_issuer(monkeypatch):
    monkeypatch.setenv("WORKER_JOB_TOKEN_SECRET", "test-secret")
    with pytest.raises(worker_auth.WorkerAuthError) as error:
        worker_auth.verify_worker_token(_token("test-secret", jti="iss-1", iss="other"), "tts")
    assert error.value.status_code == 401


def test_worker_token_rejects_revoked_jti(monkeypatch):
    monkeypatch.setenv("WORKER_JOB_TOKEN_SECRET", "test-secret")
    monkeypatch.setenv("WORKER_REVOKED_JTIS", "revoked-1")
    with pytest.raises(worker_auth.WorkerAuthError) as error:
        worker_auth.verify_worker_token(_token("test-secret", jti="revoked-1"), "tts")
    assert error.value.status_code == 401


def test_worker_token_binds_request_project():
    claims = {"job_id": "job-1", "project_id": "project-1", "tenant_id": "tenant-1", "user_id": "user-1"}
    worker_auth.assert_request_binding(claims, {"project_id": "project-1"})
    with pytest.raises(worker_auth.WorkerAuthError) as error:
        worker_auth.assert_request_binding(claims, {"project_id": "project-2"})
    assert error.value.status_code == 403
