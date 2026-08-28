import yaml

from tcc_pipeline.config import load_config, project_root_from_config, resolve_path


def test_config_paths_and_mlflow_artifact_uri_are_portable(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\nversion='0.0.0'\n", encoding="utf-8")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "project.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "test"},
                "tracking": {
                    "uri": "sqlite:///mlflow.db",
                    "artifact_root": "mlflow-artifacts:",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("TCC_MLFLOW_TRACKING_URI", raising=False)
    cfg = load_config(config_file)
    assert project_root_from_config(config_file) == tmp_path
    assert cfg["tracking"]["uri"] == f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    assert cfg["tracking"]["artifact_root"] == "mlflow-artifacts:"
    assert resolve_path(tmp_path, "data/input") == (tmp_path / "data/input").resolve()
