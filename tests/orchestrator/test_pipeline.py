import pytest
from unittest.mock import Mock, patch
import torch

from core.entities.types import AgentResponse, Message, TaskBatch
from core.entities import ConversationBatch, TaskType
from modules.rag.vector_db.base import VectorDatabaseBase
from orchestrator.pipeline import Pipeline

TEST_DEVICE = 'cpu'

@pytest.fixture(autouse=True)
def patch_pipeline_dependencies():
    with patch('orchestrator.pipeline.pipeline.ImageGenerator'), \
         patch('orchestrator.pipeline.pipeline.LLM'), \
         patch('orchestrator.pipeline.pipeline.RAG'), \
         patch('orchestrator.pipeline.pipeline.Router'), \
         patch('orchestrator.pipeline.pipeline.Embedder'):
        yield

@pytest.fixture
def mock_router_model():
    return Mock(spec=torch.nn.Module)

@pytest.fixture
def mock_vector_db():
    return Mock(spec=VectorDatabaseBase)

@pytest.fixture
def sample_conversation_batch():
    return ConversationBatch(
        histories=[
            [Message(role="user", content="What is AI?")],
            [Message(role="user", content="Generate an image of a cat")]
        ],
        conv_ids=["conv1", "conv2"]
    )


@pytest.fixture
def sample_task_batch():
    return TaskBatch(
        task=TaskType.RAG,
        histories=[[Message(role="user", content="What is AI?")]],
        conv_ids=["conv1"],
        embeddings=torch.tensor([[0.1, 0.2, 0.3]])
    )

def test_pipeline_singleton_behavior(mock_router_model, mock_vector_db):
    Pipeline._instance = None
    
    pipeline1 = Pipeline(mock_router_model, mock_vector_db)
    pipeline2 = Pipeline(mock_router_model, mock_vector_db)
    
    assert pipeline1 is pipeline2
    assert Pipeline._instance is pipeline1


def test_pipeline_initialization_once(mock_router_model, mock_vector_db):
    Pipeline._instance = None
    
    with patch('orchestrator.pipeline.pipeline.Embedder') as mock_embedder, \
         patch('orchestrator.pipeline.pipeline.Router') as mock_router, \
         patch('orchestrator.pipeline.pipeline.RAG') as mock_rag, \
         patch('orchestrator.pipeline.pipeline.LLM') as mock_llm, \
         patch('orchestrator.pipeline.pipeline.ImageGenerator') as mock_img_gen:
        
        pipeline = Pipeline(mock_router_model, mock_vector_db)
        
        assert mock_embedder.called
        assert mock_router.called
        assert mock_rag.called
        assert mock_llm.called
        assert mock_img_gen.called
        
        mock_embedder.reset_mock()
        mock_router.reset_mock()
        mock_rag.reset_mock()
        mock_llm.reset_mock()
        mock_img_gen.reset_mock()
        
        Pipeline(mock_router_model, mock_vector_db)
        
        assert not mock_embedder.called
        assert not mock_router.called
        assert not mock_rag.called
        assert not mock_llm.called
        assert not mock_img_gen.called


def test_pipeline_call_success(mock_router_model, mock_vector_db, sample_conversation_batch):
    Pipeline._instance = None
    
    with patch.object(Pipeline, '_route_and_group') as mock_route, \
         patch.object(Pipeline, '_process_task_batches') as mock_process:
        
        mock_route.return_value = {}
        mock_process.return_value = [AgentResponse(conv_id="conv1", type="text", content="response")]
        
        pipeline = Pipeline(mock_router_model, mock_vector_db)
        result = pipeline(sample_conversation_batch)
        
        mock_route.assert_called_once_with(sample_conversation_batch)
        mock_process.assert_called_once_with({})
        assert len(result) == 1
        assert result[0].conv_id == "conv1"


def test_pipeline_call_exception_handling(mock_router_model, mock_vector_db, sample_conversation_batch):
    Pipeline._instance = None
    
    with patch.object(Pipeline, '_route_and_group') as mock_route:
        mock_route.side_effect = Exception("Test error")
        
        pipeline = Pipeline(mock_router_model, mock_vector_db)
        result = pipeline(sample_conversation_batch)
        
        assert result == []


@patch('orchestrator.pipeline.pipeline.PromptTransformer')
@patch('orchestrator.pipeline.pipeline.group_conversations_by_task')
def test_route_and_group(mock_group_convs, mock_prompt_transformer, mock_router_model, mock_vector_db, sample_conversation_batch):
    Pipeline._instance = None
    
    mock_prompt_transformer.format_messages_to_str_batch.return_value = ["prompt1", "prompt2"]
    mock_embeddings = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    mock_task_types = [TaskType.RAG, TaskType.IMAGE_GEN]
    mock_probs = torch.tensor([0.8, 0.9])
    mock_grouped = {TaskType.RAG: Mock()}
    
    pipeline = Pipeline(mock_router_model, mock_vector_db)
    pipeline.embedder.extract_embeddings = Mock(return_value=mock_embeddings)
    pipeline.router.route = Mock(return_value=(mock_task_types, mock_probs))
    mock_group_convs.return_value = mock_grouped
    
    result = pipeline._route_and_group(sample_conversation_batch)
    
    mock_prompt_transformer.format_messages_to_str_batch.assert_called_once_with(sample_conversation_batch.histories)
    pipeline.embedder.extract_embeddings.assert_called_once_with(["prompt1", "prompt2"])
    pipeline.router.route.assert_called_once_with(mock_embeddings, return_probs=True)
    mock_group_convs.assert_called_once_with(
        messages=sample_conversation_batch.histories,
        task_types=mock_task_types,
        conv_ids=sample_conversation_batch.conv_ids,
        embeddings=mock_embeddings
    )
    assert result == mock_grouped


def test_process_task_batches_rag(mock_router_model, mock_vector_db, sample_task_batch):
    Pipeline._instance = None
    
    grouped_convs = {TaskType.RAG: sample_task_batch}
    expected_responses = [AgentResponse(conv_id="conv1", type="text", content="response")]
    
    with patch.object(Pipeline, '_handle_rag_task') as mock_handle_rag:
        mock_handle_rag.return_value = expected_responses
        
        pipeline = Pipeline(mock_router_model, mock_vector_db)
        result = pipeline._process_task_batches(grouped_convs)
        
        mock_handle_rag.assert_called_once_with(sample_task_batch)
        assert result == expected_responses


def test_process_task_batches_image_gen(mock_router_model, mock_vector_db, sample_task_batch):
    Pipeline._instance = None
    
    grouped_convs = {TaskType.IMAGE_GEN: sample_task_batch}
    expected_responses = [AgentResponse(conv_id="conv1", type="image", content="base64data")]
    
    with patch.object(Pipeline, '_handle_img_gen_task') as mock_handle_img:
        mock_handle_img.return_value = expected_responses
        
        pipeline = Pipeline(mock_router_model, mock_vector_db)
        result = pipeline._process_task_batches(grouped_convs)
        
        mock_handle_img.assert_called_once_with(sample_task_batch)
        assert result == expected_responses


def test_process_task_batches_unsupported_task(mock_router_model, mock_vector_db, sample_task_batch):
    Pipeline._instance = None
    
    unsupported_task = Mock()
    unsupported_task.name = "UNSUPPORTED"
    grouped_convs = {unsupported_task: sample_task_batch}
    
    pipeline = Pipeline(mock_router_model, mock_vector_db)
    result = pipeline._process_task_batches(grouped_convs)
    
    assert result == []


def test_handle_rag_task_no_stream(mock_router_model, mock_vector_db, sample_task_batch):
    Pipeline._instance = None
    
    pipeline = Pipeline(mock_router_model, mock_vector_db, llm_stream_output=False)
    pipeline.rag.add_context = Mock()
    pipeline.llm.generate_batch = Mock(return_value=["response1"])
    
    result = pipeline._handle_rag_task(sample_task_batch)
    
    pipeline.rag.add_context.assert_called_once_with(sample_task_batch.histories, sample_task_batch.embeddings)
    pipeline.llm.generate_batch.assert_called_once_with(sample_task_batch.histories)
    assert len(result) == 1
    assert result[0].conv_id == "conv1"
    assert result[0].type == "text"
    assert result[0].content == "response1"


def test_handle_rag_task_with_stream(mock_router_model, mock_vector_db, sample_task_batch):
    Pipeline._instance = None
    
    pipeline = Pipeline(mock_router_model, mock_vector_db, llm_stream_output=True)
    pipeline.rag.add_context = Mock()
    pipeline.llm.generate_stream_batch = Mock(return_value=["stream_response1"])
    
    result = pipeline._handle_rag_task(sample_task_batch)
    
    pipeline.rag.add_context.assert_called_once_with(sample_task_batch.histories, sample_task_batch.embeddings)
    pipeline.llm.generate_stream_batch.assert_called_once_with(sample_task_batch.histories)
    assert len(result) == 1
    assert result[0].conv_id == "conv1"
    assert result[0].type == "stream"
    assert result[0].content == "stream_response1"


@patch('orchestrator.pipeline.pipeline.PromptTransformer')
@patch('orchestrator.pipeline.pipeline.tensor_to_base64')
def test_handle_img_gen_task(mock_tensor_to_base64, mock_prompt_transformer, mock_router_model, mock_vector_db, sample_task_batch):
    Pipeline._instance = None
    
    mock_user_messages = [Message(role="user", content="Generate cat")]
    mock_prompts = ["Generate cat"]
    mock_img_tensor = torch.tensor([1, 2, 3])
    mock_base64 = "base64_image_data"
    
    mock_prompt_transformer.get_messages_by_role.return_value = mock_user_messages
    mock_prompt_transformer.get_content_from_messages.return_value = mock_prompts
    mock_tensor_to_base64.return_value = mock_base64
    
    pipeline = Pipeline(mock_router_model, mock_vector_db)
    pipeline.image_generator.generate = Mock(return_value=[mock_img_tensor])
    
    result = pipeline._handle_img_gen_task(sample_task_batch)
    
    mock_prompt_transformer.get_messages_by_role.assert_called_once_with(sample_task_batch.histories, role='user')
    mock_prompt_transformer.get_content_from_messages.assert_called_once_with(mock_user_messages)
    pipeline.image_generator.generate.assert_called_once_with(prompts=mock_prompts)
    mock_tensor_to_base64.assert_called_once_with(mock_img_tensor)
    
    assert len(result) == 1
    assert result[0].conv_id == "conv1"
    assert result[0].type == "image"
    assert result[0].content == mock_base64


@patch('orchestrator.pipeline.factory.PipelineFactory')
def test_build_classmethod(mock_factory):
    Pipeline._instance = None
    mock_pipeline = Mock()
    mock_factory.build.return_value = mock_pipeline
    
    result = Pipeline.build()
    
    mock_factory.build.assert_called_once()
    assert result == mock_pipeline


def test_pipeline_parameters_initialization(mock_router_model, mock_vector_db):
    Pipeline._instance = None
    
    with patch('orchestrator.pipeline.pipeline.Embedder') as mock_embedder, \
         patch('orchestrator.pipeline.pipeline.RAG') as mock_rag, \
         patch('orchestrator.pipeline.pipeline.LLM') as mock_llm:
        
        pipeline = Pipeline(
            router_model=mock_router_model,
            rag_vector_db=mock_vector_db,
            rag_n_extracted_docs=10,
            embed_model_name='custom-embedder',
            llm_model_name='custom-llm',
            llm_temperature=0.5
        )
        
        mock_embedder.assert_called_once_with(model_name='custom-embedder', device_name='cuda')
        mock_rag.assert_called_once_with(
            vector_db=mock_vector_db,
            n_extracted_docs=10,
            prompt_format='{context}\n{prompt}'
        )
        mock_llm.assert_called_once_with(
            model_name='custom-llm',
            dtype='int8',
            max_new_tokens=1024,
            temperature=0.5,
            device_name='cuda'
        )
        assert pipeline.stream_output == False


def test_empty_grouped_conversations(mock_router_model, mock_vector_db):
    Pipeline._instance = None
    
    pipeline = Pipeline(mock_router_model, mock_vector_db)
    result = pipeline._process_task_batches({})
    
    assert result == []


def test_multiple_task_types_processing(mock_router_model, mock_vector_db):
    Pipeline._instance = None
    
    rag_batch = TaskBatch(
        task=TaskType.RAG,
        histories=[[Message(role="user", content="What is AI?")]],
        conv_ids=["conv1"],
        embeddings=torch.tensor([[0.1, 0.2]])
    )
    img_batch = TaskBatch(
        task=TaskType.IMAGE_GEN,
        histories=[[Message(role="user", content="Generate cat")]],
        conv_ids=["conv2"],
        embeddings=torch.tensor([[0.3, 0.4]])
    )
    
    grouped_convs = {
        TaskType.RAG: rag_batch,
        TaskType.IMAGE_GEN: img_batch
    }
    
    with patch.object(Pipeline, '_handle_rag_task') as mock_handle_rag, \
         patch.object(Pipeline, '_handle_img_gen_task') as mock_handle_img:
        
        mock_handle_rag.return_value = [AgentResponse(conv_id="conv1", type="text", content="rag_response")]
        mock_handle_img.return_value = [AgentResponse(conv_id="conv2", type="image", content="img_data")]
        
        pipeline = Pipeline(mock_router_model, mock_vector_db)
        result = pipeline._process_task_batches(grouped_convs)
        
        assert len(result) == 2
        assert any(r.conv_id == "conv1" and r.type == "text" for r in result)
        assert any(r.conv_id == "conv2" and r.type == "image" for r in result)