import pytest
import torch
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from core.entities.enums import ModelDType
from core.entities.types import Message
from modules.llm.engines.transformers import TransformersEngine
from modules.llm.configs import LLMEngineConfig, LLMGenerationConfig
from modules.llm.components import TransformersTextStreamer

TEST_MODEL_NAME = "gpt2"
TEST_DEVICE = "cpu"


@pytest.fixture
def mock_model():
    """Provide mocked transformers model."""
    mock_model = Mock()
    mock_model.device = torch.device(TEST_DEVICE)
    mock_model.eval.return_value = mock_model
    return mock_model


@pytest.fixture
def mock_tokenizer():
    """Provide mocked tokenizer."""
    mock_tokenizer = Mock()
    mock_tokenizer.pad_token = "<pad>"
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    return mock_tokenizer


@pytest.fixture
def mock_messages():
    """Provide test messages."""
    return [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!")
    ]


@pytest.fixture
def engine_config():
    """Provide test engine config."""
    config = LLMEngineConfig(
        model_name=TEST_MODEL_NAME,
        dtype=ModelDType.FLOAT16
    )
    config.extra_data['device'] = TEST_DEVICE
    return config


def test_init(mock_model, mock_tokenizer):
    """Test TransformersEngine initialization."""
    with patch('modules.llm.engines.transformers.TransformersTextStreamer') as mock_streamer:
        engine = TransformersEngine(mock_model, mock_tokenizer)
        
        assert engine.model is mock_model
        assert engine.tokenizer is mock_tokenizer
        assert engine._device == mock_model.device
        mock_streamer.assert_called_once_with(mock_model, mock_tokenizer)


@patch('modules.llm.engines.transformers.get_torch_device')
def test_from_config_success(mock_get_device, engine_config):
    """Test successful engine creation from config."""
    mock_device = torch.device(TEST_DEVICE)
    mock_get_device.return_value = mock_device
    
    with patch.object(TransformersEngine, 'load_model') as mock_load_model, \
         patch.object(TransformersEngine, 'load_tokenizer') as mock_load_tokenizer, \
         patch('modules.llm.engines.transformers.TransformersTextStreamer'):
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = mock_model
        mock_load_tokenizer.return_value = mock_tokenizer
        
        result = TransformersEngine.from_config(engine_config)
        
        assert isinstance(result, TransformersEngine)
        mock_get_device.assert_called_once_with('cpu')
        mock_load_model.assert_called_once_with(TEST_MODEL_NAME, ModelDType.FLOAT16, mock_device)
        mock_load_tokenizer.assert_called_once_with(TEST_MODEL_NAME)


def test_from_config_failure(engine_config):
    """Test engine creation failure from config."""
    with patch.object(TransformersEngine, 'load_model', side_effect=Exception("Load failed")):
        
        with pytest.raises(Exception):
            TransformersEngine.from_config(engine_config)


@patch('shared.config.Config.get')
def test_load_model_local_path_exists(mock_config_get):
    """Test loading model from local path when it exists."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=True), \
         patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_from_pretrained, \
         patch('modules.llm.engines.transformers.to_torch_dtype', return_value=torch.float16):
        
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_from_pretrained.return_value = mock_model
        
        result = TransformersEngine.load_model(
            TEST_MODEL_NAME,
            ModelDType.FLOAT16,
            torch.device(TEST_DEVICE)
        )
        
        assert result is mock_model
        mock_from_pretrained.assert_called_once_with(
            f"/tmp/models/{TEST_MODEL_NAME}",
            quantization_config=None,
            torch_dtype=torch.float16,
            device_map=torch.device(TEST_DEVICE),
            local_files_only=True
        )
        mock_model.eval.assert_called_once()


@patch('shared.config.Config.get')
def test_load_model_from_hub(mock_config_get):
    """Test loading model from HuggingFace Hub when local doesn't exist."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=False), \
         patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_from_pretrained, \
         patch('modules.llm.engines.transformers.to_torch_dtype', return_value=torch.float16):
        
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_from_pretrained.return_value = mock_model
        
        result = TransformersEngine.load_model(
            TEST_MODEL_NAME,
            ModelDType.FLOAT16,
            torch.device(TEST_DEVICE)
        )
        
        assert result is mock_model
        mock_from_pretrained.assert_called_once_with(
            TEST_MODEL_NAME,
            quantization_config=None,
            torch_dtype=torch.float16,
            device_map=torch.device(TEST_DEVICE)
        )


@patch('shared.config.Config.get')
def test_load_model_with_quantization(mock_config_get):
    """Test loading model with quantization config."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=False), \
         patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_from_pretrained, \
         patch('modules.llm.engines.transformers.get_bitsandbytes_config_for_dtype') as mock_quant_config:
        
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_from_pretrained.return_value = mock_model
        mock_config = Mock()
        mock_quant_config.return_value = mock_config
        
        result = TransformersEngine.load_model(
            TEST_MODEL_NAME,
            ModelDType.INT8,
            torch.device(TEST_DEVICE)
        )
        
        assert result is mock_model
        mock_quant_config.assert_called_once_with(ModelDType.INT8)
        mock_from_pretrained.assert_called_once_with(
            TEST_MODEL_NAME,
            quantization_config=mock_config,
            torch_dtype=None,
            device_map=torch.device(TEST_DEVICE)
        )


@patch('shared.config.Config.get')
def test_load_model_failure(mock_config_get):
    """Test model loading failure."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=False), \
         patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=Exception("Load failed")):
        
        with pytest.raises(Exception):
            TransformersEngine.load_model(
                TEST_MODEL_NAME,
                ModelDType.FLOAT16,
                torch.device(TEST_DEVICE)
            )


@patch('shared.config.Config.get')
def test_load_tokenizer_local_path_exists(mock_config_get):
    """Test loading tokenizer from local path when it exists."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=True), \
         patch('transformers.AutoTokenizer.from_pretrained') as mock_from_pretrained:
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_from_pretrained.return_value = mock_tokenizer
        
        result = TransformersEngine.load_tokenizer(TEST_MODEL_NAME)
        
        assert result is mock_tokenizer
        mock_from_pretrained.assert_called_once_with(
            f"/tmp/models/{TEST_MODEL_NAME}",
            local_files_only=True
        )


@patch('shared.config.Config.get')
def test_load_tokenizer_from_hub(mock_config_get):
    """Test loading tokenizer from HuggingFace Hub when local doesn't exist."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=False), \
         patch('transformers.AutoTokenizer.from_pretrained') as mock_from_pretrained:
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_from_pretrained.return_value = mock_tokenizer
        
        result = TransformersEngine.load_tokenizer(TEST_MODEL_NAME)
        
        assert result is mock_tokenizer
        mock_from_pretrained.assert_called_once_with(TEST_MODEL_NAME)


@patch('shared.config.Config.get')
def test_load_tokenizer_sets_pad_token(mock_config_get):
    """Test that tokenizer pad_token is set if None."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=False), \
         patch('transformers.AutoTokenizer.from_pretrained') as mock_from_pretrained:
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_from_pretrained.return_value = mock_tokenizer
        
        result = TransformersEngine.load_tokenizer(TEST_MODEL_NAME)
        
        assert result.pad_token == "<eos>"


@patch('shared.config.Config.get')
def test_load_tokenizer_failure(mock_config_get):
    """Test tokenizer loading failure."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=False), \
         patch('transformers.AutoTokenizer.from_pretrained', side_effect=Exception("Load failed")):
        
        with pytest.raises(Exception):
            TransformersEngine.load_tokenizer(TEST_MODEL_NAME)


def test_create_generator(mock_model, mock_tokenizer, mock_messages):
    """Test create_generator delegates to streamer."""
    with patch('modules.llm.engines.transformers.TransformersTextStreamer') as mock_streamer_class:
        mock_streamer = Mock()
        mock_generator = AsyncMock()
        mock_streamer.create_generator.return_value = mock_generator
        mock_streamer_class.return_value = mock_streamer
        
        engine = TransformersEngine(mock_model, mock_tokenizer)
        config = LLMGenerationConfig(temperature=0.8)
        
        result = engine.create_generator(mock_messages, config, custom_param="test")
        
        assert result is mock_generator
        mock_streamer.create_generator.assert_called_once_with(
            mock_messages, config, custom_param="test"
        )


def test_create_batch_generators(mock_model, mock_tokenizer):
    """Test create_batch_generators delegates to streamer with validation."""
    messages_batch = [
        [Message(role="user", content="Hello")],
        [Message(role="user", content="Hi")]
    ]
    configs = [
        LLMGenerationConfig(temperature=0.7),
        LLMGenerationConfig(temperature=0.9)
    ]
    
    with patch('modules.llm.engines.transformers.TransformersTextStreamer') as mock_streamer_class:
        mock_streamer = Mock()
        mock_generators = [AsyncMock(), AsyncMock()]
        mock_streamer.create_batch_generators.return_value = mock_generators
        mock_streamer_class.return_value = mock_streamer
        
        engine = TransformersEngine(mock_model, mock_tokenizer)
        
        # Mock the validation method
        with patch.object(engine, '_validate_batch_parameters', return_value={}):
            result = engine.create_batch_generators(messages_batch, configs)
            
            assert result == mock_generators
            mock_streamer.create_batch_generators.assert_called_once_with(
                messages_batch, configs
            )


def test_create_batch_generators_with_validation(mock_model, mock_tokenizer):
    """Test create_batch_generators includes validated parameters."""
    messages_batch = [
        [Message(role="user", content="Hello")],
        [Message(role="user", content="Hi")]
    ]
    
    batch_params = {
        "temperatures": [0.7, 0.9],
        "max_tokens": [100, 200]
    }
    
    validated_params = {
        "temperatures": [0.7, 0.9]
    }
    
    with patch('modules.llm.engines.transformers.TransformersTextStreamer') as mock_streamer_class:
        mock_streamer = Mock()
        mock_generators = [AsyncMock(), AsyncMock()]
        mock_streamer.create_batch_generators.return_value = mock_generators
        mock_streamer_class.return_value = mock_streamer
        
        engine = TransformersEngine(mock_model, mock_tokenizer)
        
        # Mock the validation method to return filtered params
        with patch.object(engine, '_validate_batch_parameters', return_value=validated_params):
            result = engine.create_batch_generators(messages_batch, None, **batch_params)
            
            assert result == mock_generators
            mock_streamer.create_batch_generators.assert_called_once_with(
                messages_batch, None, **validated_params
            )


def test_device_property(mock_model, mock_tokenizer):
    """Test that engine exposes model device."""
    with patch('modules.llm.engines.transformers.TransformersTextStreamer'):
        engine = TransformersEngine(mock_model, mock_tokenizer)
        assert engine._device == mock_model.device


def test_quantization_dtype_mapping():
    """Test that quantization and regular dtypes are handled correctly."""
    device = torch.device("cpu")
    
    # Test INT4 quantization
    with patch('modules.llm.engines.transformers.get_bitsandbytes_config_for_dtype') as mock_quant, \
         patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_from_pretrained, \
         patch.object(Path, 'exists', return_value=False):
        
        mock_config = Mock()
        mock_quant.return_value = mock_config
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_from_pretrained.return_value = mock_model
        
        TransformersEngine.load_model("test", ModelDType.INT4, device)
        
        mock_quant.assert_called_once_with(ModelDType.INT4)
        mock_from_pretrained.assert_called_once_with(
            "test",
            quantization_config=mock_config,
            torch_dtype=None,
            device_map=device
        )
    
    # Test regular dtype
    with patch('modules.llm.engines.transformers.to_torch_dtype', return_value=torch.float32) as mock_dtype, \
         patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_from_pretrained, \
         patch.object(Path, 'exists', return_value=False):
        
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_from_pretrained.return_value = mock_model
        
        TransformersEngine.load_model("test", ModelDType.FLOAT32, device)
        
        mock_dtype.assert_called_once_with(ModelDType.FLOAT32)
        mock_from_pretrained.assert_called_once_with(
            "test",
            quantization_config=None,
            torch_dtype=torch.float32,
            device_map=device
        )