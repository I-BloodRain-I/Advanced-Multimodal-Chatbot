import pytest
import uuid
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
from core.entities.enums import ModelDType
from core.entities.types import Message
from modules.llm.engines.vllm import VLLMEngine
from modules.llm.configs import LLMEngineConfig, LLMGenerationConfig

TEST_MODEL_NAME = "gpt2"


@pytest.fixture
def mock_vllm_model():
    """Provide mocked vLLM model."""
    mock_model = Mock(spec=AsyncLLMEngine)
    mock_model.generate = AsyncMock()
    return mock_model


@pytest.fixture
def mock_tokenizer():
    """Provide mocked tokenizer."""
    mock_tokenizer = Mock(spec=AutoTokenizer)
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
    config.extra_data.update({
        "quantization": "fp8",
        "tensor_parallel_size": 1
    })
    return config


def test_init(mock_vllm_model, mock_tokenizer):
    """Test VLLMEngine initialization."""
    engine = VLLMEngine(mock_vllm_model, mock_tokenizer)
    
    assert engine.model is mock_vllm_model
    assert engine.tokenizer is mock_tokenizer


def test_from_config_success(engine_config):
    """Test successful engine creation from config."""
    with patch.object(VLLMEngine, 'load_model') as mock_load_model, \
         patch.object(VLLMEngine, 'load_tokenizer') as mock_load_tokenizer:
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = mock_model
        mock_load_tokenizer.return_value = mock_tokenizer
        
        result = VLLMEngine.from_config(engine_config)
        
        assert isinstance(result, VLLMEngine)
        mock_load_model.assert_called_once_with(
            TEST_MODEL_NAME,
            ModelDType.FLOAT16,
            quantization="fp8",
            tensor_parallel_size=1
        )
        mock_load_tokenizer.assert_called_once_with(TEST_MODEL_NAME)


def test_from_config_failure(engine_config):
    """Test engine creation failure from config."""
    with patch.object(VLLMEngine, 'load_model', side_effect=Exception("Load failed")):
        
        with pytest.raises(Exception):
            VLLMEngine.from_config(engine_config)


@patch('modules.llm.engines.base.Config.get')
def test_load_model_local_path_exists(mock_config_get):
    """Test loading model from local path when it exists."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=True), \
         patch('modules.llm.engines.vllm.AsyncEngineArgs') as mock_args, \
         patch('modules.llm.engines.vllm.AsyncLLMEngine.from_engine_args') as mock_from_args:
        
        mock_engine = Mock()
        mock_from_args.return_value = mock_engine
        
        result = VLLMEngine.load_model(
            TEST_MODEL_NAME,
            ModelDType.FLOAT16,
            quantization="fp8"
        )
        
        assert result is mock_engine
        mock_args.assert_called_once_with(
            model=f"/tmp/models/{TEST_MODEL_NAME}",
            dtype="float16",
            quantization="fp8"
        )
        mock_from_args.assert_called_once_with(mock_args.return_value)


@patch('modules.llm.engines.base.Config.get')
def test_load_model_from_hub(mock_config_get):
    """Test loading model from HuggingFace Hub when local doesn't exist."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=False), \
         patch('modules.llm.engines.vllm.AsyncEngineArgs') as mock_args, \
         patch('modules.llm.engines.vllm.AsyncLLMEngine.from_engine_args') as mock_from_args:
        
        mock_engine = Mock()
        mock_from_args.return_value = mock_engine
        
        result = VLLMEngine.load_model(
            TEST_MODEL_NAME,
            ModelDType.BFLOAT16,
            tensor_parallel_size=2
        )
        
        assert result is mock_engine
        mock_args.assert_called_once_with(
            model=TEST_MODEL_NAME,
            dtype="bfloat16",
            tensor_parallel_size=2
        )


def test_load_model_dtype_mapping():
    """Test that ModelDType enums are correctly mapped to vLLM dtypes."""
    dtype_mappings = [
        (ModelDType.AUTO, "auto"),
        (ModelDType.INT4, "auto"),  # vLLM doesn't support int4 directly
        (ModelDType.INT8, "auto"),  # vLLM doesn't support int8 directly
        (ModelDType.BFLOAT16, "bfloat16"),
        (ModelDType.FLOAT16, "float16"),
        (ModelDType.FLOAT32, "float32")
    ]
    
    for model_dtype, expected_vllm_dtype in dtype_mappings:
        with patch.object(Path, 'exists', return_value=False), \
             patch('modules.llm.engines.vllm.AsyncEngineArgs') as mock_args, \
             patch('modules.llm.engines.vllm.AsyncLLMEngine.from_engine_args'):
            
            VLLMEngine.load_model("test", model_dtype)
            
            mock_args.assert_called_once_with(
                model="test",
                dtype=expected_vllm_dtype
            )


@patch('modules.llm.engines.base.Config.get')
def test_load_model_failure(mock_config_get):
    """Test model loading failure."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=False), \
         patch('modules.llm.engines.vllm.AsyncEngineArgs'), \
         patch('vllm.AsyncLLMEngine.from_engine_args', side_effect=Exception("Load failed")):
        
        with pytest.raises(Exception):
            VLLMEngine.load_model(TEST_MODEL_NAME, ModelDType.FLOAT16)


@patch('modules.llm.engines.base.Config.get')
def test_load_tokenizer_local_path_exists(mock_config_get):
    """Test loading tokenizer from local path when it exists."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=True), \
         patch('modules.llm.engines.vllm.AutoTokenizer.from_pretrained') as mock_from_pretrained:
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_from_pretrained.return_value = mock_tokenizer
        
        result = VLLMEngine.load_tokenizer(TEST_MODEL_NAME)
        
        assert result is mock_tokenizer
        mock_from_pretrained.assert_called_once_with(
            f"/tmp/models/{TEST_MODEL_NAME}",
            local_files_only=True
        )


@patch('modules.llm.engines.base.Config.get')
def test_load_tokenizer_from_hub(mock_config_get):
    """Test loading tokenizer from HuggingFace Hub when local doesn't exist."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=False), \
         patch('modules.llm.engines.vllm.AutoTokenizer.from_pretrained') as mock_from_pretrained:
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_from_pretrained.return_value = mock_tokenizer
        
        result = VLLMEngine.load_tokenizer(TEST_MODEL_NAME)
        
        assert result is mock_tokenizer
        mock_from_pretrained.assert_called_once_with(TEST_MODEL_NAME)


@patch('modules.llm.engines.base.Config.get')
def test_load_tokenizer_sets_pad_token(mock_config_get):
    """Test that tokenizer pad_token is set if None."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=False), \
         patch('modules.llm.engines.vllm.AutoTokenizer.from_pretrained') as mock_from_pretrained:
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_from_pretrained.return_value = mock_tokenizer
        
        result = VLLMEngine.load_tokenizer(TEST_MODEL_NAME)
        
        assert result.pad_token == "<eos>"


@patch('modules.llm.engines.base.Config.get')
def test_load_tokenizer_failure(mock_config_get):
    """Test tokenizer loading failure."""
    mock_config_get.return_value = "/tmp/models"
    
    with patch.object(Path, 'exists', return_value=False), \
         patch('transformers.AutoTokenizer.from_pretrained', side_effect=Exception("Load failed")):
        
        with pytest.raises(Exception):
            VLLMEngine.load_tokenizer(TEST_MODEL_NAME)


def test_create_sampling_params():
    """Test sampling parameters creation."""
    engine = VLLMEngine.__new__(VLLMEngine)
    
    # Test with max_new_tokens mapping
    params1 = engine._create_sampling_params(
        temperature=0.8,
        max_new_tokens=100,
        top_p=0.9
    )
    
    assert isinstance(params1, SamplingParams)
    # Note: We can't easily test the internal values without accessing private attributes
    
    # Test without max_new_tokens
    params2 = engine._create_sampling_params(
        temperature=0.7,
        top_p=0.95
    )
    
    assert isinstance(params2, SamplingParams)


@pytest.mark.asyncio
async def test_extract_token_stream():
    """Test token extraction from vLLM output stream."""
    engine = VLLMEngine.__new__(VLLMEngine)
    
    # Mock RequestOutput objects
    output1 = Mock(spec=RequestOutput)
    output1.outputs = [Mock()]
    output1.outputs[0].text = "Hello"
    
    output2 = Mock(spec=RequestOutput)
    output2.outputs = [Mock()]
    output2.outputs[0].text = "Hello world"
    
    output3 = Mock(spec=RequestOutput)
    output3.outputs = [Mock()]
    output3.outputs[0].text = "Hello world!"
    
    # Create async generator
    async def mock_generator():
        yield output1
        yield output2
        yield output3
    
    # Test token extraction
    tokens = []
    async for token in engine._extract_token_stream(mock_generator()):
        tokens.append(token)
    
    assert tokens == ["Hello", " world", "!"]


@pytest.mark.asyncio
async def test_extract_token_stream_error():
    """Test token extraction with malformed output."""
    engine = VLLMEngine.__new__(VLLMEngine)
    
    # Mock malformed RequestOutput
    output = Mock(spec=RequestOutput)
    output.outputs = []  # Empty outputs list
    
    async def mock_generator():
        yield output
    
    with pytest.raises(IndexError):
        async for _ in engine._extract_token_stream(mock_generator()):
            pass


def test_create_generator_empty_messages(mock_vllm_model, mock_tokenizer):
    """Test create_generator with empty messages raises ValueError."""
    engine = VLLMEngine(mock_vllm_model, mock_tokenizer)
    
    with pytest.raises(ValueError, match="Messages cannot be empty"):
        engine.create_generator([])


@patch('modules.llm.engines.vllm.PromptTransformer.format_messages_to_str')
@patch('modules.llm.engines.vllm.uuid.uuid4')
def test_create_generator_success(mock_uuid, mock_format, mock_vllm_model, mock_tokenizer, mock_messages):
    """Test successful generator creation."""
    mock_uuid.return_value = "test-request-id"
    mock_format.return_value = "formatted prompt"
    
    engine = VLLMEngine(mock_vllm_model, mock_tokenizer)
    config = LLMGenerationConfig(temperature=0.8, max_tokens=100, priority=5)
    
    with patch.object(engine, '_create_sampling_params') as mock_sampling, \
         patch.object(engine, '_extract_token_stream') as mock_extract:
        
        mock_sampling_params = Mock()
        mock_sampling.return_value = mock_sampling_params
        mock_generator = Mock()
        mock_vllm_model.generate.return_value = mock_generator
        mock_extract_generator = AsyncMock()
        mock_extract.return_value = mock_extract_generator
        
        result = engine.create_generator(mock_messages, config)
        
        assert result is mock_extract_generator
        mock_format.assert_called_once_with(mock_messages, mock_tokenizer)
        mock_sampling.assert_called_once_with(
            temperature=0.8,
            max_new_tokens=100,
            top_p=0.9  # default value
        )
        mock_vllm_model.generate.assert_called_once_with(
            "formatted prompt",
            sampling_params=mock_sampling_params,
            request_id="test-request-id",
            priority=5
        )
        mock_extract.assert_called_once()


@patch('modules.llm.engines.vllm.PromptTransformer.format_messages_to_str')
def test_create_generator_without_config(mock_format, mock_vllm_model, mock_tokenizer, mock_messages):
    """Test generator creation without config uses defaults."""
    mock_format.return_value = "formatted prompt"
    
    engine = VLLMEngine(mock_vllm_model, mock_tokenizer)
    
    with patch.object(engine, '_create_sampling_params') as mock_sampling, \
         patch.object(engine, '_extract_token_stream'):
        
        mock_sampling_params = Mock()
        mock_sampling.return_value = mock_sampling_params
        
        engine.create_generator(mock_messages)
        
        # Should use default config values
        mock_sampling.assert_called_once_with(
            temperature=0.8,
            top_p=0.9,
            max_new_tokens=1024
        )


def test_create_batch_generators_empty_batch(mock_vllm_model, mock_tokenizer):
    """Test create_batch_generators with empty batch raises ValueError."""
    engine = VLLMEngine(mock_vllm_model, mock_tokenizer)
    
    with pytest.raises(ValueError, match="Messages batch cannot be empty"):
        engine.create_batch_generators([])


def test_create_batch_generators_config_length_mismatch(mock_vllm_model, mock_tokenizer):
    """Test create_batch_generators with mismatched configs length."""
    messages_batch = [
        [Message(role="user", content="Hello")],
        [Message(role="user", content="Hi")]
    ]
    configs = [LLMGenerationConfig()]  # Only one config for two messages
    
    engine = VLLMEngine(mock_vllm_model, mock_tokenizer)
    
    with pytest.raises(ValueError, match="Configs length \\(1\\) must match batch size \\(2\\)"):
        engine.create_batch_generators(messages_batch, configs)


def test_create_batch_generators_success(mock_vllm_model, mock_tokenizer):
    """Test successful batch generator creation."""
    messages_batch = [
        [Message(role="user", content="Hello")],
        [Message(role="user", content="Hi")]
    ]
    configs = [
        LLMGenerationConfig(temperature=0.7),
        LLMGenerationConfig(temperature=0.9)
    ]
    
    engine = VLLMEngine(mock_vllm_model, mock_tokenizer)
    
    with patch.object(engine, 'create_generator') as mock_create, \
         patch.object(engine, '_validate_batch_parameters', return_value={}):
        
        mock_generators = [AsyncMock(), AsyncMock()]
        mock_create.side_effect = mock_generators
        
        result = engine.create_batch_generators(messages_batch, configs)
        
        assert result == mock_generators
        assert mock_create.call_count == 2
        
        # Check that each generator was created with correct parameters
        for i, call in enumerate(mock_create.call_args_list):
            assert call[1]['messages'] == messages_batch[i]  # messages
            assert call[1]['config'] == configs[i]  # config


def test_create_batch_generators_with_validation(mock_vllm_model, mock_tokenizer):
    """Test batch generator creation includes validated parameters."""
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
    
    engine = VLLMEngine(mock_vllm_model, mock_tokenizer)
    
    with patch.object(engine, 'create_generator') as mock_create, \
         patch.object(engine, '_validate_batch_parameters', return_value=validated_params):
        
        mock_generators = [AsyncMock(), AsyncMock()]
        mock_create.side_effect = mock_generators
        
        result = engine.create_batch_generators(messages_batch, None, **batch_params)
        
        assert result == mock_generators
        engine._validate_batch_parameters.assert_called_once_with(messages_batch, **batch_params)
        
        # Check that validated parameters are passed to each generator
        for i, call in enumerate(mock_create.call_args_list):
            # Should get temperature from validated params
            assert "temperatures" not in call[1]  # Individual params extracted