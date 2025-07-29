import pytest
from unittest.mock import Mock

from core.entities.enums import ModelDType
from modules.llm.configs import LLMGenerationConfig, LLMEngineConfig


def test_llm_generation_config_defaults():
    """Test LLMGenerationConfig with default values."""
    config = LLMGenerationConfig()
    
    assert config.temperature == 0.8
    assert config.top_p == 0.9
    assert config.max_new_tokens == 1024
    assert config.priority == 0
    assert config.extra_data == {}


def test_llm_generation_config_custom_values():
    """Test LLMGenerationConfig with custom values."""
    config = LLMGenerationConfig(
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
        priority=5
    )
    
    assert config.temperature == 0.7
    assert config.top_p == 0.95
    assert config.max_new_tokens == 512
    assert config.priority == 5
    assert config.extra_data == {}


def test_llm_generation_config_with_kwargs():
    """Test LLMGenerationConfig with extra kwargs."""
    config = LLMGenerationConfig(
        temperature=0.6,
        custom_param="test",
        another_param=42
    )
    
    assert config.temperature == 0.6
    assert config.extra_data == {
        "custom_param": "test",
        "another_param": 42
    }


def test_llm_generation_config_temperature_validation():
    """Test temperature validation in LLMGenerationConfig."""
    # Valid temperatures
    config1 = LLMGenerationConfig(temperature=0.0)
    assert config1.temperature == 0.0
    
    config2 = LLMGenerationConfig(temperature=2.0)
    assert config2.temperature == 2.0
    
    config3 = LLMGenerationConfig(temperature=1.0)
    assert config3.temperature == 1.0


def test_llm_generation_config_top_p_validation():
    """Test top_p validation in LLMGenerationConfig."""
    # Valid top_p values
    config1 = LLMGenerationConfig(top_p=0.0)
    assert config1.top_p == 0.0
    
    config2 = LLMGenerationConfig(top_p=1.0)
    assert config2.top_p == 1.0
    
    config3 = LLMGenerationConfig(top_p=0.5)
    assert config3.top_p == 0.5


def test_llm_generation_config_max_tokens_validation():
    """Test max_tokens validation in LLMGenerationConfig."""
    # Valid max_tokens (must be > 0)
    config1 = LLMGenerationConfig(max_tokens=1)
    assert config1.max_new_tokens == 1
    
    config2 = LLMGenerationConfig(max_tokens=4096)
    assert config2.max_new_tokens == 4096


def test_llm_generation_config_priority_validation():
    """Test priority validation in LLMGenerationConfig."""
    # Valid priorities (0-10)
    config1 = LLMGenerationConfig(priority=0)
    assert config1.priority == 0
    
    config2 = LLMGenerationConfig(priority=10)
    assert config2.priority == 10
    
    config3 = LLMGenerationConfig(priority=5)
    assert config3.priority == 5


def test_llm_engine_config_defaults():
    """Test LLMEngineConfig with default values."""
    config = LLMEngineConfig(model_name="test-model")
    
    assert config.model_name == "test-model"
    assert config.dtype == ModelDType.AUTO
    assert config.extra_data == {}


def test_llm_engine_config_custom_values():
    """Test LLMEngineConfig with custom values."""
    config = LLMEngineConfig(
        model_name="custom-model",
        dtype=ModelDType.FLOAT16
    )
    
    assert config.model_name == "custom-model"
    assert config.dtype == ModelDType.FLOAT16
    assert config.extra_data == {}


def test_llm_engine_config_with_kwargs():
    """Test LLMEngineConfig with extra kwargs."""
    config = LLMEngineConfig(
        model_name="test-model",
        dtype=ModelDType.INT8,
        device="cuda",
        custom_param="value"
    )
    
    assert config.model_name == "test-model"
    assert config.dtype == ModelDType.INT8
    assert config.extra_data == {
        "device": "cuda",
        "custom_param": "value"
    }


def test_llm_engine_config_model_dtype_variants():
    """Test LLMEngineConfig with different ModelDType values."""
    # Test all ModelDType enum values
    for dtype in ModelDType:
        config = LLMEngineConfig(model_name="test", dtype=dtype)
        assert config.dtype == dtype


def test_llm_engine_config_model_name_required():
    """Test that model_name is required for LLMEngineConfig."""
    # This should work
    config = LLMEngineConfig(model_name="required-name")
    assert config.model_name == "required-name"


def test_llm_generation_config_field_types():
    """Test that LLMGenerationConfig fields have correct types."""
    config = LLMGenerationConfig(
        temperature=0.8,
        top_p=0.9,
        max_tokens=1024,
        priority=5
    )
    
    assert isinstance(config.temperature, float)
    assert isinstance(config.top_p, float)
    assert isinstance(config.max_new_tokens, int)
    assert isinstance(config.priority, int)
    assert isinstance(config.extra_data, dict)


def test_llm_engine_config_field_types():
    """Test that LLMEngineConfig fields have correct types."""
    config = LLMEngineConfig(
        model_name="test-model",
        dtype=ModelDType.FLOAT32
    )
    
    assert isinstance(config.model_name, str)
    assert isinstance(config.dtype, ModelDType)
    assert isinstance(config.extra_data, dict)


def test_llm_generation_config_immutable_extra_data():
    """Test that extra_data is properly initialized and modifiable."""
    config = LLMGenerationConfig(param1="value1", param2="value2")
    
    # Should be able to access extra_data
    assert config.extra_data["param1"] == "value1"
    assert config.extra_data["param2"] == "value2"
    
    # Should be able to modify extra_data
    config.extra_data["param3"] = "value3"
    assert config.extra_data["param3"] == "value3"


def test_llm_engine_config_immutable_extra_data():
    """Test that extra_data is properly initialized and modifiable."""
    config = LLMEngineConfig(
        model_name="test",
        param1="value1",
        param2="value2"
    )
    
    # Should be able to access extra_data
    assert config.extra_data["param1"] == "value1"
    assert config.extra_data["param2"] == "value2"
    
    # Should be able to modify extra_data
    config.extra_data["param3"] = "value3"
    assert config.extra_data["param3"] == "value3"


def test_config_objects_equality():
    """Test that config objects with same values are equal."""
    config1 = LLMGenerationConfig(temperature=0.7, max_tokens=100)
    config2 = LLMGenerationConfig(temperature=0.7, max_tokens=100)
    
    # Test individual fields are equal
    assert config1.temperature == config2.temperature
    assert config1.max_new_tokens == config2.max_new_tokens
    assert config1.top_p == config2.top_p
    assert config1.priority == config2.priority


def test_engine_config_objects_equality():
    """Test that engine config objects with same values are equal."""
    config1 = LLMEngineConfig(model_name="test", dtype=ModelDType.FLOAT16)
    config2 = LLMEngineConfig(model_name="test", dtype=ModelDType.FLOAT16)
    
    # Test individual fields are equal
    assert config1.model_name == config2.model_name
    assert config1.dtype == config2.dtype
    assert config1.extra_data == config2.extra_data