import pytest
from unittest.mock import Mock
from shared.text_processing import PromptTransformer


def create_mock_message(role, content):
    message = Mock()
    message.role = role
    message.content = content
    message.model_dump = Mock(return_value={"role": role, "content": content})
    return message


def test_concat_messages_with_histories():
    message = create_mock_message("user", "Hello")
    history = []
    result = PromptTransformer.concat_messages_with_histories(message, history)
    
    assert len(result) == 1
    assert result[0] == message


def test_concat_messages_with_histories_empty_message():
    history = [Mock()]
    result = PromptTransformer.concat_messages_with_histories(None, history)
    
    assert result == history


def test_concat_messages_with_histories_batch():
    messages = [create_mock_message("user", "Hello"), create_mock_message("user", "Hi")]
    histories = [[], []]
    
    result = PromptTransformer.concat_messages_with_histories_batch(messages, histories)
    
    assert len(result) == 2
    assert result[0] == [messages[0]]
    assert result[1] == [messages[1]]


def test_concat_messages_with_histories_batch_mismatch():
    with pytest.raises(ValueError, match="Mismatch: 1 messages vs 2 histories"):
        PromptTransformer.concat_messages_with_histories_batch([Mock()], [[], []])


def test_format_messages_to_str_with_tokenizer():
    messages = [create_mock_message("user", "Hello")]
    tokenizer = Mock()
    tokenizer.chat_template = True
    tokenizer.apply_chat_template = Mock(return_value="formatted prompt")
    
    result = PromptTransformer.format_messages_to_str(messages, tokenizer)
    
    assert result == "formatted prompt"
    tokenizer.apply_chat_template.assert_called_once()


def test_format_messages_to_str_without_tokenizer():
    messages = [
        create_mock_message("user", "Hello"),
        create_mock_message("assistant", "Hi there"),
        create_mock_message("system", "System message")
    ]
    
    result = PromptTransformer.format_messages_to_str(messages)
    
    assert result == "User: Hello\nAssistant: Hi there\nSystem: System message"


@pytest.mark.parametrize("messages", [[], None])
def test_format_messages_to_str_empty(messages):
    assert PromptTransformer.format_messages_to_str(messages) == ""


def test_format_messages_to_str_batch():
    messages_batch = [
        [create_mock_message("user", "Hello")],
        [create_mock_message("user", "Hi")]
    ]
    
    result = PromptTransformer.format_messages_to_str_batch(messages_batch)
    
    assert result == ["User: Hello", "User: Hi"]


def test_get_messages_by_role():
    messages_batch = [
        [create_mock_message("user", "Hello"), create_mock_message("assistant", "Hi")],
        [create_mock_message("user", "Test")]
    ]
    
    result = PromptTransformer.get_messages_by_role(messages_batch, "user")
    
    assert len(result) == 2
    assert len(result[0]) == 1
    assert result[0][0].role == "user"
    assert result[1][0].role == "user"


def test_get_messages_by_role_invalid():
    with pytest.raises(Exception):
        PromptTransformer.get_messages_by_role([[]], "invalid")


@pytest.mark.parametrize("separator,expected", [
    ("\n", "Hello\nWorld"),
    (" | ", "Hello | World"),
    ("", "HelloWorld")
])
def test_get_content_from_messages(separator, expected):
    messages_batch = [[
        create_mock_message("user", "Hello"),
        create_mock_message("user", "World")
    ]]
    
    result = PromptTransformer.get_content_from_messages(messages_batch, separator)
    
    assert result[0] == expected


def test_convert_messages_to_dict():
    messages_batch = [
        [create_mock_message("user", "Hello")],
        [create_mock_message("assistant", "Hi")]
    ]
    
    result = PromptTransformer.convert_messages_to_dict(messages_batch)
    
    expected = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "assistant", "content": "Hi"}]
    ]
    assert result == expected