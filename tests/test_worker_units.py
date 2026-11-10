import ast
import os
import pathlib
import subprocess
import tempfile
import types
from typing import Any, Dict, List, Optional

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"
SOURCE = MAIN.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class DummyModel:
    pass


def load_functions(*names):
    wanted = set(names)
    nodes = []
    for node in TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            node.decorator_list = []
            nodes.append(node)
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "__builtins__": __builtins__,
        "os": os,
        "subprocess": subprocess,
        "tempfile": tempfile,
        "requests": types.SimpleNamespace(),
        "HTTPException": HTTPException,
        "DEFAULT_TG_BOT_TOKEN": "",
        "DEFAULT_TG_CHAT_ID": "",
        "DEFAULT_STREAM_BRIDGE": "https://bridge.test",
        "TelegramUploadConfig": DummyModel,
        "TtsSynthesisRequest": DummyModel,
        "FfmpegEpisodeRequest": DummyModel,
        "List": List,
        "Dict": Dict,
        "Any": Any,
        "Optional": Optional,
    }
    exec(compile(module, str(MAIN), "exec"), namespace)
    return namespace


def function_source(name):
    node = next(
        item for item in TREE.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(SOURCE, node)


def schema_fields(name):
    node = next(
        item for item in TREE.body
        if isinstance(item, ast.ClassDef) and item.name == name
    )
    fields = {}
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            fields[statement.target.id] = statement.value is not None
    return fields


def test_tts_pipeline_rejects_blank_voice_text():
    namespace = load_functions("run_tts_pipeline")
    for text in ("", "   ", "\n\t"):
        with pytest.raises(HTTPException) as error:
            namespace["run_tts_pipeline"](types.SimpleNamespace(voice_text=text))
        assert error.value.status_code == 400


def test_render_episode_rejects_empty_scene_manifest():
    namespace = load_functions("render_episode")
    with pytest.raises(HTTPException) as error:
        namespace["render_episode"](types.SimpleNamespace(scenes_manifest=[]))
    assert error.value.status_code == 400


def test_ass_time_formatting_handles_boundaries():
    namespace = load_functions("format_ass_time")
    formatter = namespace["format_ass_time"]
    assert formatter(0) == "0:00:00.00"
    assert formatter(65.5) == "0:01:05.50"
    assert formatter(3599.999) == "0:59:59.99"
    assert formatter(3661.25) == "1:01:01.25"


def test_ass_subtitle_writer_skips_blank_and_escapes_newlines(tmp_path):
    namespace = load_functions("format_ass_time", "generate_ass_subtitle_file")
    target = tmp_path / "subs.ass"
    namespace["generate_ass_subtitle_file"](
        [
            {"start": 0.1, "end": 2.0, "text": "Hello\nworld"},
            {"start": 2.0, "end": 3.0, "text": "   "},
        ],
        str(target),
    )
    output = target.read_text(encoding="utf-8")
    assert "[Script Info]" in output
    assert "Dialogue: 0,0:00:00.10,0:00:02.00,Default,,0,0,0,,Hello\\Nworld" in output
    assert output.count("Dialogue:") == 1


def test_probe_duration_parses_value_and_falls_back_safely():
    namespace = load_functions("probe_duration")
    calls = []

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return "12.5\n"

    namespace["subprocess"] = types.SimpleNamespace(check_output=check_output)
    assert namespace["probe_duration"]("voice.wav") == 12.5
    assert calls[0][1]["timeout"] == 15

    namespace["subprocess"] = types.SimpleNamespace(check_output=lambda *a, **k: "0.1\n")
    assert namespace["probe_duration"]("voice.wav", 7.0) == 7.0

    def failing(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffprobe", 15)

    namespace["subprocess"] = types.SimpleNamespace(check_output=failing)
    assert namespace["probe_duration"]("voice.wav", 4.0) == 4.0


class FakeResponse:
    def __init__(self, ok=True, chunks=(b"abc",), payload=None, error=None):
        self.ok = ok
        self.chunks = chunks
        self.payload = payload or {}
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_content(self, chunk_size):
        assert chunk_size > 0
        for chunk in self.chunks:
            yield chunk

    def json(self):
        return self.payload


def test_download_file_rejects_unsafe_urls_and_reports_failures(tmp_path):
    namespace = load_functions("download_file")
    called = []

    def get(url, **kwargs):
        called.append((url, kwargs))
        return FakeResponse(ok=True, chunks=(b"data",))

    namespace["requests"] = types.SimpleNamespace(get=get)
    target = tmp_path / "asset.bin"
    assert namespace["download_file"]("file:///etc/passwd", str(target)) is False
    assert called == []
    assert namespace["download_file"]("https://cdn.test/a.bin", str(target)) is True
    assert target.read_bytes() == b"data"
    assert called[0][1]["timeout"] == 60

    namespace["requests"] = types.SimpleNamespace(get=lambda *a, **k: FakeResponse(ok=False))
    assert namespace["download_file"]("https://cdn.test/b.bin", str(target)) is False

    def failing(*args, **kwargs):
        raise OSError("network down")

    namespace["requests"] = types.SimpleNamespace(get=failing)
    assert namespace["download_file"]("https://cdn.test/c.bin", str(target)) is False


def test_upload_to_telegram_returns_null_without_credentials_or_file(tmp_path):
    namespace = load_functions("upload_to_telegram")
    target = tmp_path / "audio.wav"
    target.write_bytes(b"audio")
    called = []
    namespace["requests"] = types.SimpleNamespace(post=lambda *a, **k: called.append((a, k)))
    assert namespace["upload_to_telegram"](str(target), "caption", None) == {
        "file_id": None,
        "telegram_url": None,
    }
    assert called == []
    assert namespace["upload_to_telegram"](str(tmp_path / "missing.wav"), "caption", None) == {
        "file_id": None,
        "telegram_url": None,
    }


def test_upload_to_telegram_retries_and_maps_file_id(tmp_path):
    namespace = load_functions("upload_to_telegram")
    namespace["DEFAULT_TG_BOT_TOKEN"] = "bot-token"
    namespace["DEFAULT_TG_CHAT_ID"] = "chat-id"
    target = tmp_path / "audio.wav"
    target.write_bytes(b"audio")
    attempts = []

    def post(url, **kwargs):
        attempts.append((url, kwargs))
        if len(attempts) < 3:
            raise OSError("temporary")
        return FakeResponse(payload={"result": {"document": {"file_id": "file-1"}}})

    namespace["requests"] = types.SimpleNamespace(post=post)
    result = namespace["upload_to_telegram"](str(target), "caption", None)
    assert len(attempts) == 3
    assert attempts[0][1]["timeout"] == 180
    assert result == {
        "file_id": "file-1",
        "telegram_url": "https://bridge.test/file-1",
    }


def test_upload_to_telegram_returns_null_after_retry_exhaustion(tmp_path):
    namespace = load_functions("upload_to_telegram")
    namespace["DEFAULT_TG_BOT_TOKEN"] = "bot-token"
    namespace["DEFAULT_TG_CHAT_ID"] = "chat-id"
    target = tmp_path / "audio.wav"
    target.write_bytes(b"audio")
    attempts = []

    def post(*args, **kwargs):
        attempts.append(kwargs)
        raise OSError("permanent")

    namespace["requests"] = types.SimpleNamespace(post=post)
    assert namespace["upload_to_telegram"](str(target), "caption", None) == {
        "file_id": None,
        "telegram_url": None,
    }
    assert len(attempts) == 3


def test_worker_schema_required_fields_match_workflow_contracts():
    tts = schema_fields("TtsSynthesisRequest")
    for field in ("project_id", "episode_no", "scene_order", "voice_text"):
        assert tts[field] is False
    assert tts["provider"] is True
    assert tts["telegram_upload"] is True

    ffmpeg = schema_fields("FfmpegEpisodeRequest")
    for field in ("project_id", "episode_no", "scenes_manifest"):
        assert ffmpeg[field] is False
    for field in ("engine", "burn_subtitles", "telegram_upload"):
        assert ffmpeg[field] is True

    scene = schema_fields("SceneManifestItem")
    assert scene["scene_order"] is False
    assert scene["duration_sec"] is True
    assert scene["sfx_tracks"] is True

    assert schema_fields("SfxTrackItem")["sfx_code"] is True
    assert schema_fields("TelegramUploadConfig")["enabled"] is True


def test_worker_retry_timeout_and_cleanup_contracts_are_present():
    upload = function_source("upload_to_telegram")
    assert "range(3)" in upload
    assert "timeout=180" in upload

    download = function_source("download_file")
    assert "timeout=60" in download

    probe = function_source("probe_duration")
    assert "timeout=15" in probe

    render = function_source("render_episode")
    assert "timeout=600" in render
    assert "tempfile.TemporaryDirectory()" in render

    tts = function_source("run_tts_pipeline")
    assert "tempfile.TemporaryDirectory()" in tts
    assert "timeout=60" in tts


def test_health_endpoint_contract():
    namespace = load_functions("health")
    payload = namespace["health"]()
    assert payload["ok"] is True
    assert payload["worker"] == "github-cpu-worker"
    assert "Svara" in payload["tts_engine"]
