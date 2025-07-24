import os
import yaml
import pytest
from shared.config import Config

@pytest.fixture(scope="function", autouse=True)
def isolated_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    yield str(config_path)

def test_default_config_creation(isolated_config):
    Config._instance = None
    config = Config()
    assert os.path.exists(isolated_config), "config.yaml should be created"
    assert config.get("models_dir") == "models"

def test_config_get_set_update(isolated_config):
    config = Config()

    # Test set
    config.set("llm.temperature", 0.9)
    assert config.get("llm.temperature") == 0.9

    # Test get fallback
    assert config.get("non.existent.key", default="fallback") == "fallback"

    # Test update
    config.update({"custom_key": "value"})
    assert config.get("custom_key") == "value"

def test_config_reload_and_save(isolated_config):
    Config._instance = None
    config = Config()
    config.set("llm.model_name", "new-model")
    config.save()

    # Read directly from file to ensure it was saved
    with open(isolated_config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
        assert raw["llm"]["model_name"] == "new-model"

    # Modify file and reload
    raw["llm"]["model_name"] = "reloaded-model"
    with open(isolated_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f)

    config.reload()
    assert config.get("llm.model_name") == "reloaded-model"
    config._instance = None

def test_config_validation_failure(monkeypatch, isolated_config):
    Config._instance = None
    monkeypatch.setenv("CONFIG_PATH", isolated_config)
    config = Config()

    with open(isolated_config, "w", encoding="utf-8") as f:
        yaml.safe_dump({"llm": {}}, f)  # Incomplete config

    with pytest.raises(ValueError, match="Missing required config key"):
        Config.reload()
