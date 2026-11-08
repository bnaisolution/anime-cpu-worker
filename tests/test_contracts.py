import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"
REQUIREMENTS = ROOT / "requirements.txt"


def tree():
    return ast.parse(MAIN.read_text(encoding="utf-8"))


def decorator_routes(tree_root):
    routes = set()
    for node in ast.walk(tree_root):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in {"get", "post", "put", "delete", "patch"}:
                continue
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                routes.add((func.attr.upper(), decorator.args[0].value))
    return routes


def test_worker_routes_are_present():
    routes = decorator_routes(tree())
    expected = {
        ("POST", "/v1/tts/svara"),
        ("POST", "/v1/tts/supertonic"),
        ("POST", "/v1/ffmpeg/render-episode"),
        ("GET", "/health"),
    }
    assert expected <= routes


def test_worker_schemas_are_defined():
    classes = {node.name for node in ast.walk(tree()) if isinstance(node, ast.ClassDef)}
    expected = {
        "TelegramUploadConfig",
        "TtsSynthesisRequest",
        "SfxTrackItem",
        "SceneManifestItem",
        "FfmpegEpisodeRequest",
    }
    assert expected <= classes


def test_worker_has_no_literal_credentials():
    source = MAIN.read_text(encoding="utf-8")
    patterns = [
        r"TELEGRAM_BOT_TOKEN\s*=\s*[\"'][0-9]+:[A-Za-z0-9_-]+[\"']",
        r"AKIA[0-9A-Z]{16}",
        r"ghp_[A-Za-z0-9]{20,}",
        r"sk-[A-Za-z0-9]{20,}",
    ]
    for pattern in patterns:
        assert re.search(pattern, source) is None, pattern


def test_worker_secrets_are_environment_driven():
    source = MAIN.read_text(encoding="utf-8")
    assert 'os.getenv("TELEGRAM_BOT_TOKEN"' in source
    assert 'os.getenv("TELEGRAM_STORAGE_CHAT_ID"' in source


def test_worker_requirements_are_pinned():
    lines = [
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    unpinned = [line for line in lines if not re.fullmatch(r"[A-Za-z0-9_.-]+==[^=]+", line)]
    assert unpinned == []
