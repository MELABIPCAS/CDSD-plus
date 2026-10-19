from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import project_config


def test_config_defaults_are_repository_relative(monkeypatch, tmp_path):
    for name in ("CDSD_WORKSPACE_DATA", "CDSD_PROJECT_DATA", "CDSD_INFO_XLSX", "CDSD_OUTPUT_ROOT", "CDSD_CONFIG_FILE"):
        monkeypatch.delenv(name, raising=False)
    # Isolate defaults from the user's real config/paths.local.json.
    monkeypatch.setattr(project_config, "LOCAL_CONFIG", tmp_path / "paths.local.json")
    config = project_config.get_config()
    assert config.repo_root == Path(__file__).resolve().parents[1]
    assert config.output_root == config.repo_root
    assert isinstance(config.workspace_data, Path)
