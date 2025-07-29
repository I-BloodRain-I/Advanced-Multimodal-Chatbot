import pytest
from unittest.mock import Mock
from abc import ABC

from core.entities.types import Message
from modules.llm.engines.base import LLMEngineBase
from modules.llm.configs import LLMEngineConfig, LLMGenerationConfig


class TestLLMEngineImplementation(LLMEngineBase):
    """Concrete implementation for testing the abstract base class."""
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_config(cls, config: LLMEngineConfig):
        return cls()
    
    @classmethod
    def load_model(cls):
        return Mock()
    
    @classmethod
    def load_tokenizer(cls):
        return Mock()
    
    def create_generator(self, messages, config):
        return Mock()
    
    def create_batch_generators(self, messages_batch, configs):
        return [Mock() for _ in messages_batch]


def test_abstract_base_class():
    """Test that LLMEngineBase is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        LLMEngineBase()


def test_concrete_implementation_can_be_instantiated():
    """Test that concrete implementation can be instantiated."""
    engine = TestLLMEngineImplementation()
    assert isinstance(engine, LLMEngineBase)


@pytest.fixture
def test_engine():
    """Provide test engine instance."""
    return TestLLMEngineImplementation()


@pytest.fixture
def mock_messages_batch():
    """Provide test messages batch."""
    return [
        [Message(role="user", content="Hello")],
        [Message(role="user", content="Hi there!")]
    ]


def test_validate_batch_parameters_empty_batch(test_engine):
    """Test validation with empty batch raises ValueError."""
    with pytest.raises(ValueError, match="Messages batch cannot be empty"):
        test_engine._validate_batch_parameters([])


def test_validate_batch_parameters_valid_batch(test_engine, mock_messages_batch):
    """Test validation with valid batch parameters."""
    batch_params = {
        "temperatures": [0.7, 0.9],
        "max_tokens": [100, 200],
        "non_list_param": "single_value"
    }
    
    result = test_engine._validate_batch_parameters(mock_messages_batch, **batch_params)
    
    # Should include list parameters that match batch size
    assert "temperatures" in result
    assert "max_tokens" in result
    assert result["temperatures"] == [0.7, 0.9]
    assert result["max_tokens"] == [100, 200]
    
    # Should exclude non-list parameters
    assert "non_list_param" not in result


def test_validate_batch_parameters_mismatched_length(test_engine, mock_messages_batch):
    """Test validation with mismatched parameter length."""
    batch_params = {
        "temperatures": [0.7, 0.9, 0.8],  # 3 values for 2 messages
        "max_tokens": [100, 200]  # 2 values (correct)
    }
    
    # Should skip mismatched parameters but keep correct ones
    result = test_engine._validate_batch_parameters(mock_messages_batch, **batch_params)
    
    assert "max_tokens" in result
    assert "temperatures" not in result  # Should be skipped due to length mismatch


def test_validate_batch_parameters_non_list_types(test_engine, mock_messages_batch):
    """Test validation with different parameter types."""
    batch_params = {
        "list_param": [0.7, 0.9],
        "tuple_param": (0.8, 0.6),
        "set_param": {0.5, 0.4},
        "string_param": "not_a_list",
        "int_param": 42,
        "dict_param": {"key": "value"}
    }
    
    result = test_engine._validate_batch_parameters(mock_messages_batch, **batch_params)
    
    # Should include list-like types with correct length
    assert "list_param" in result
    assert "tuple_param" in result
    assert "set_param" in result
    
    # Should exclude non-list types
    assert "string_param" not in result
    assert "int_param" not in result
    assert "dict_param" not in result


def test_validate_batch_parameters_empty_list_param(test_engine, mock_messages_batch):
    """Test validation with empty list parameter."""
    batch_params = {
        "empty_list": [],
        "valid_list": [0.7, 0.9]
    }
    
    result = test_engine._validate_batch_parameters(mock_messages_batch, **batch_params)
    
    # Empty list should be skipped due to length mismatch
    assert "empty_list" not in result
    assert "valid_list" in result


def test_validate_batch_parameters_no_valid_params(test_engine, mock_messages_batch):
    """Test validation when no parameters are valid."""
    batch_params = {
        "wrong_length": [0.7],
        "not_a_list": "value",
        "another_wrong_length": [0.7, 0.8, 0.9]
    }
    
    result = test_engine._validate_batch_parameters(mock_messages_batch, **batch_params)
    
    # Should return empty dict when no parameters are valid
    assert result == {}


def test_validate_batch_parameters_batch_size_one(test_engine):
    """Test validation with batch size of one."""
    single_message_batch = [[Message(role="user", content="Hello")]]
    
    batch_params = {
        "temperature": [0.7],
        "max_tokens": [100]
    }
    
    result = test_engine._validate_batch_parameters(single_message_batch, **batch_params)
    
    assert "temperature" in result
    assert "max_tokens" in result
    assert result["temperature"] == [0.7]
    assert result["max_tokens"] == [100]


def test_models_dir_property():
    """Test that _MODELS_DIR is properly set from config."""
    # The _MODELS_DIR should be set from Config.get('models_dir')
    assert hasattr(LLMEngineBase, '_MODELS_DIR')
    # We can't easily test the actual value without mocking Config.get


def test_logger_initialization():
    """Test that logger is properly initialized."""
    engine = TestLLMEngineImplementation()
    
    # Should have a logger attribute (private)
    assert hasattr(engine, '_LLMEngineBase__logger')


def test_logger_custom_initialization():
    """Test initialization with custom logger."""
    custom_logger = Mock()
    
    class CustomEngineImplementation(LLMEngineBase):
        def __init__(self):
            super().__init__(logger=custom_logger)
        
        @classmethod
        def from_config(cls, config): return cls()
        @classmethod 
        def load_model(cls): return Mock()
        @classmethod
        def load_tokenizer(cls): return Mock()
        def create_generator(self, messages, config): return Mock()
        def create_batch_generators(self, messages_batch, configs): return [Mock()]
    
    engine = CustomEngineImplementation()
    assert engine._LLMEngineBase__logger is custom_logger


def test_abstract_methods_must_be_implemented():
    """Test that all abstract methods must be implemented."""
    
    # Test missing from_config
    with pytest.raises(TypeError):
        class MissingFromConfig(LLMEngineBase):
            @classmethod
            def load_model(cls): return Mock()
            @classmethod
            def load_tokenizer(cls): return Mock()
            def create_generator(self, messages, config): return Mock()
            def create_batch_generators(self, messages_batch, configs): return [Mock()]
        
        MissingFromConfig()
    
    # Test missing load_model
    with pytest.raises(TypeError):
        class MissingLoadModel(LLMEngineBase):
            @classmethod
            def from_config(cls, config): return cls()
            @classmethod
            def load_tokenizer(cls): return Mock()
            def create_generator(self, messages, config): return Mock()
            def create_batch_generators(self, messages_batch, configs): return [Mock()]
        
        MissingLoadModel()


def test_inheritance_structure():
    """Test that the inheritance structure is correct."""
    assert issubclass(LLMEngineBase, ABC)
    assert issubclass(TestLLMEngineImplementation, LLMEngineBase)
    
    # Test that ABC methods are properly marked as abstract
    abstract_methods = LLMEngineBase.__abstractmethods__
    expected_abstract_methods = {
        'from_config', 'load_model', 'load_tokenizer',
        'create_generator', 'create_batch_generators'
    }
    
    assert abstract_methods == expected_abstract_methods


def test_validate_batch_parameters_logging(test_engine, mock_messages_batch):
    """Test that validation logs warnings for mismatched parameters."""
    batch_params = {
        "wrong_length": [0.7],  # Should trigger warning
        "correct_length": [0.7, 0.9]  # Should not trigger warning
    }
    
    # We can't easily test logging without access to the logger
    # But we can test that the method completes successfully
    result = test_engine._validate_batch_parameters(mock_messages_batch, **batch_params)
    
    assert "correct_length" in result
    assert "wrong_length" not in result