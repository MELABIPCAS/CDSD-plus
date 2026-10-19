"""Path configuration for the reproducible acoustic-analysis package.

The repository contains code only. Private data locations are supplied through
environment variables or an ignored JSON file, so the scripts do not depend on
the original author's drive letters.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config" / "paths.example.json"
LOCAL_CONFIG = REPO_ROOT / "config" / "paths.local.json"


@dataclass(frozen=True)
class ProjectConfig:
    repo_root: Path
    workspace_data: Path
    project_data: Path
    info_xlsx: Path
    output_root: Path


def _read_local_config() -> dict[str, str]:
    configured = os.environ.get("CDSD_CONFIG_FILE")
    path = Path(configured).expanduser() if configured else LOCAL_CONFIG
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return {str(key): str(value) for key, value in payload.items() if value is not None}


def _resolve(name: str, local: dict[str, str], default: Path) -> Path:
    value = os.environ.get(name) or local.get(name)
    return Path(value).expanduser() if value else default


def get_config() -> ProjectConfig:
    local = _read_local_config()
    return ProjectConfig(
        repo_root=REPO_ROOT,
        workspace_data=_resolve("CDSD_WORKSPACE_DATA", local, REPO_ROOT / "data"),
        project_data=_resolve("CDSD_PROJECT_DATA", local, REPO_ROOT / "data"),
        info_xlsx=_resolve("CDSD_INFO_XLSX", local, REPO_ROOT / "data" / "CDSD_Plus_100名被试最新信息总表.xlsx"),
        output_root=_resolve("CDSD_OUTPUT_ROOT", local, REPO_ROOT),
    )


def require_inputs(config: ProjectConfig) -> None:
    """Raise an actionable error when a data-dependent stage lacks inputs."""
    missing = [
        ("CDSD_WORKSPACE_DATA", config.workspace_data),
        ("CDSD_PROJECT_DATA", config.project_data),
        ("CDSD_INFO_XLSX", config.info_xlsx),
    ]
    absent = [f"{name}={path}" for name, path in missing if not path.exists()]
    if absent:
        raise FileNotFoundError(
            "Private analysis inputs are missing. Configure them with environment variables "
            "or config/paths.local.json:\n" + "\n".join(absent)
        )

