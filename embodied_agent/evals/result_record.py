from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

RESULT_SCHEMA_VERSION = 1


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "unknown"


def git_revision(path: str | Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(path)), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    revision = completed.stdout.strip()
    return revision or None


def package_vcs_revision(distribution_name: str) -> str | None:
    try:
        raw = metadata.distribution(distribution_name).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    try:
        direct_url = json.loads(raw)
    except json.JSONDecodeError:
        return None
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, Mapping):
        return None
    commit_id = vcs_info.get("commit_id")
    return str(commit_id) if commit_id else None


def physics_environment_metadata(
    *, xlerobot_runtime_root: str | Path, humanoid_runtime_root: str | Path
) -> dict[str, Any]:
    humanoid_root = Path(humanoid_runtime_root)
    humanoid_model_root = humanoid_root / "robot" / "lerobot-humanoid-model"
    return {
        "repository_revision": os.getenv("GITHUB_SHA") or git_revision(Path.cwd()),
        "python_version": _python_version(),
        "upstream_revisions": {
            "gym-pybullet-drones": package_vcs_revision("gym-pybullet-drones"),
            "xlerobot": git_revision(xlerobot_runtime_root),
            "lerobot-humanoid-runtime": git_revision(humanoid_root),
            "lerobot-humanoid-model": (
                git_revision(humanoid_model_root) if humanoid_model_root.exists() else None
            ),
        },
    }


def build_result_record(
    *,
    benchmark: str,
    provider: str,
    model: str,
    max_steps: int,
    result: Mapping[str, Any],
    environment: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    for name, value in (("benchmark", benchmark), ("provider", provider), ("model", model)):
        if not value.strip():
            raise ValueError(f"{name} must be non-empty")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "benchmark": benchmark,
        "provider": provider,
        "model": model,
        "max_steps": int(max_steps),
        "created_at": created_at or utc_timestamp(),
        "environment": dict(environment or {}),
        "result": dict(result),
    }


def default_result_path(
    *, root: str | Path, benchmark: str, model: str, created_at: str
) -> Path:
    timestamp = re.sub(r"[^0-9TZ]", "", created_at)
    return Path(root) / slugify(benchmark) / slugify(model) / f"{timestamp}.json"


def write_result_record(record: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def _python_version() -> str:
    import platform

    return platform.python_version()
