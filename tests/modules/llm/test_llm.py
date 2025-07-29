import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from core.entities.types import Message, MessageHistory
from core.entities.enums import ModelDType
from modules.llm import LLM
from modules.llm.configs import LLMEngineConfig, LLMGenerationConfig
from modules.llm.engines import TransformersEngine, VLLMEngine

# Test configuration constants
TEST_MODEL_NAME = "gpt2"
TEST_ENGINE = "transformers"


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instance before each test."""
    LLM._instance = None
    yield
    LLM._instance = None


@pytest.fixture
def mock_messages():
    """Provide test messages."""
    return [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!")
    ]


@pytest.fixture
def mock_messages_batch():
    """Provide test messages batch."""
    return [
        [Message(role="user", content="Hello")],
        [Message(role="user", content="Hi there!")]
    ]


@pytest.fixture
def mock_engine_config():
    """Provide test engine config."""
    return LLMEngineConfig(model_name=TEST_MODEL_NAME, dtype=ModelDType.FLOAT16)


@pytest.fixture
def mock_generation_config():
    """Provide test generation config."""
    return LLMGenerationConfig(temperature=0.7, max_new_tokens=50)


def test_singleton_pattern(mock_engine_config):
    """Test that LLM follows singleton pattern."""
    with patch.object(LLM, '_cache_model'), \
         patch.object(LLM, 'load_engine') as mock_load_engine:
        
        mock_engine = Mock()
        mock_load_engine.return_value = mock_engine
        
        llm1 = LLM(TEST_ENGINE, mock_engine_config)
        llm2 = LLM("vllm", mock_engine_config)  # Different params
        
        assert llm1 is llm2
        assert id(llm1) == id(llm2)
        # Should only be called once due to singleton
        assert mock_load_engine.call_count == 1


def test_singleton_initialization_once(mock_engine_config):
    """Test that singleton is initialized only once."""
    with patch.object(LLM, '_cache_model') as mock_cache, \
         patch.object(LLM, 'load_engine') as mock_load_engine:
        
        mock_engine = Mock()
        mock_load_engine.return_value = mock_engine
        
        llm1 = LLM(TEST_ENGINE, mock_engine_config)
        llm2 = LLM(TEST_ENGINE, mock_engine_config)
        
        assert mock_cache.call_count == 1
        assert mock_load_engine.call_count == 1
        assert llm1._initialized is True
        assert llm2._initialized is True


@patch('shared.config.Config.get')
def test_cache_model_creates_directory_and_saves(mock_config_get, mock_engine_config):
    """Test model caching functionality."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=False), \
         patch('modules.llm.engines.TransformersEngine.from_config') as mock_from_config:
        
        mock_engine = Mock()
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_engine.model = mock_model
        mock_engine.tokenizer = mock_tokenizer
        mock_from_config.return_value = mock_engine
        
        llm = LLM.__new__(LLM)
        llm._cache_model(TEST_MODEL_NAME)
        
        mock_from_config.assert_called_once()
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()


@patch('shared.config.Config.get')
def test_cache_model_skips_if_exists(mock_config_get, mock_engine_config):
    """Test model caching skips if model already exists."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=True), \
         patch('modules.llm.engines.TransformersEngine.from_config') as mock_from_config:
        
        llm = LLM.__new__(LLM)
        llm._cache_model(TEST_MODEL_NAME)
        
        mock_from_config.assert_not_called()


def test_load_engine_transformers():
    """Test loading transformers engine."""
    config = LLMEngineConfig(model_name=TEST_MODEL_NAME)
    
    with patch('modules.llm.engines.TransformersEngine.from_config') as mock_from_config:
        mock_engine = Mock()
        mock_from_config.return_value = mock_engine
        
        result = LLM.load_engine('transformers', config)
        
        assert result is mock_engine
        mock_from_config.assert_called_once_with(config)


def test_load_engine_vllm():
    """Test loading vllm engine."""
    config = LLMEngineConfig(model_name=TEST_MODEL_NAME)
    
    with patch('modules.llm.engines.VLLMEngine.from_config') as mock_from_config:
        mock_engine = Mock()
        mock_from_config.return_value = mock_engine
        
        result = LLM.load_engine('vllm', config)
        
        assert result is mock_engine
        mock_from_config.assert_called_once_with(config)


def test_load_engine_invalid():
    """Test loading invalid engine raises error."""
    config = LLMEngineConfig(model_name=TEST_MODEL_NAME)
    
    with pytest.raises(ValueError, match="Engine name must be either: \\[transformers, vllm\\]"):
        LLM.load_engine('invalid', config)


@patch('shared.config.Config.get')
def test_generate_with_config(mock_config_get, mock_messages, mock_generation_config):
    """Test generation with provided config."""
    mock_config_get.return_value = {"temperature": 0.8, "max_new_tokens": 100}
    
    mock_engine = Mock()
    mock_generator = AsyncMock()
    mock_engine.create_generator.return_value = mock_generator
    
    llm = LLM.__new__(LLM)
    llm.engine = mock_engine
    llm._initialized = True
    
    result = llm.generate(mock_messages, mock_generation_config)
    
    assert result is mock_generator
    mock_engine.create_generator.assert_called_once_with(mock_messages, mock_generation_config)


@patch('shared.config.Config.get')
def test_generate_without_config(mock_config_get, mock_messages):
    """Test generation without config uses default from Config."""
    generation_args = {"temperature": 0.8, "max_tokens": 100, "top_p": 0.9}
    mock_config_get.return_value = generation_args
    
    mock_engine = Mock()
    mock_generator = AsyncMock()
    mock_engine.create_generator.return_value = mock_generator
    
    llm = LLM.__new__(LLM)
    llm.engine = mock_engine
    llm._initialized = True
    
    result = llm.generate(mock_messages)
    
    assert result is mock_generator
    mock_engine.create_generator.assert_called_once()
    
    # Check that a config was created from the generation args
    call_args = mock_engine.create_generator.call_args[0]
    config_used = call_args[1]  # Second positional argument
    assert isinstance(config_used, LLMGenerationConfig)
    assert config_used.temperature == 0.8
    assert config_used.max_new_tokens == 100


@patch('shared.config.Config.get')
def test_generate_batch_with_configs(mock_config_get, mock_messages_batch):
    """Test batch generation with provided configs."""
    mock_config_get.return_value = {"temperature": 0.8, "max_new_tokens": 100}
    
    configs = [
        LLMGenerationConfig(temperature=0.7),
        LLMGenerationConfig(temperature=0.9)
    ]
    
    mock_engine = Mock()
    mock_generators = [AsyncMock(), AsyncMock()]
    mock_engine.create_batch_generators.return_value = mock_generators
    
    llm = LLM.__new__(LLM)
    llm.engine = mock_engine
    llm._initialized = True
    
    result = llm.generate_batch(mock_messages_batch, configs)
    
    assert result == mock_generators
    mock_engine.create_batch_generators.assert_called_once_with(mock_messages_batch, configs)


@patch('shared.config.Config.get')
def test_generate_batch_without_configs(mock_config_get, mock_messages_batch):
    """Test batch generation without configs creates defaults."""
    generation_args = {"temperature": 0.8, "max_tokens": 100, "top_p": 0.9}
    mock_config_get.return_value = generation_args
    
    mock_engine = Mock()
    mock_generators = [AsyncMock(), AsyncMock()]
    mock_engine.create_batch_generators.return_value = mock_generators
    
    llm = LLM.__new__(LLM)
    llm.engine = mock_engine
    llm._initialized = True
    
    result = llm.generate_batch(mock_messages_batch)
    
    assert result == mock_generators
    mock_engine.create_batch_generators.assert_called_once()
    
    # Check that configs were created
    call_args = mock_engine.create_batch_generators.call_args[0]
    configs_used = call_args[1]  # Second positional argument
    assert len(configs_used) == len(mock_messages_batch)
    assert all(isinstance(config, LLMGenerationConfig) for config in configs_used)


def test_initialization_failure_propagates(mock_engine_config):
    """Test that initialization failures are properly propagated."""
    with patch.object(LLM, '_cache_model'), \
         patch.object(LLM, 'load_engine', side_effect=Exception("Engine load failed")):
        
        with pytest.raises(Exception, match="Engine load failed"):
            LLM(TEST_ENGINE, mock_engine_config)


def test_cache_model_failure_propagates():
    """Test that cache model failures are properly propagated."""
    with patch.object(LLM, '_cache_model', side_effect=Exception("Cache failed")), \
         patch.object(LLM, 'load_engine'):
        
        config = LLMEngineConfig(model_name=TEST_MODEL_NAME)
        with pytest.raises(Exception, match="Cache failed"):
            LLM(TEST_ENGINE, config)