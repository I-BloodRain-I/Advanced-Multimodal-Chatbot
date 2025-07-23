import pytest
import torch
import numpy as np
from unittest.mock import Mock
from core.entities import TaskType
from orchestrator.router import Router


def create_mock_model(output_tensor):
    mock_model = Mock()
    mock_model.return_value = output_tensor
    mock_model.parameters = Mock(side_effect=lambda: iter([torch.tensor([1.0])]))
    return mock_model


def test_router_initialization():
    mock_model = create_mock_model(torch.tensor([[1.0, 2.0, 3.0]]))
    router = Router(mock_model)
    
    assert router.model == mock_model
    assert router.device == torch.tensor([1.0]).device


def test_route_with_numpy_embeddings():
    output = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    
    task_types, probs = router.route(embeddings, return_probs=False)
    
    assert len(task_types) == 2
    assert all(isinstance(task_type, TaskType) for task_type in task_types)
    assert probs is None
    mock_model.assert_called_once()


def test_route_with_torch_tensor_embeddings():
    output = torch.tensor([[2.0, 1.0, 3.0]])
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = torch.tensor([[0.1, 0.2, 0.3]])
    
    task_types, probs = router.route(embeddings, return_probs=False)
    
    assert len(task_types) == 1
    assert isinstance(task_types[0], TaskType)
    assert probs is None


def test_route_with_list_embeddings():
    output = torch.tensor([[1.0, 3.0, 2.0]])
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = [[0.1, 0.2, 0.3]]
    
    task_types, probs = router.route(embeddings, return_probs=False)
    
    assert len(task_types) == 1
    assert isinstance(task_types[0], TaskType)


def test_route_with_return_probs_true():
    output = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    
    task_types, probs = router.route(embeddings, return_probs=True)
    
    assert len(task_types) == 2
    assert probs is not None
    assert isinstance(probs, torch.Tensor)
    assert probs.shape == (2,)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


def test_route_with_return_probs_false():
    output = torch.tensor([[1.0, 2.0, 3.0]])
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = np.array([[0.1, 0.2, 0.3]])
    
    task_types, probs = router.route(embeddings, return_probs=False)
    
    assert probs is None


def test_route_single_embedding():
    output = torch.tensor([[2.5, 1.0, 3.0]])
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = np.array([[0.1, 0.2, 0.3]])
    
    task_types, probs = router.route(embeddings, return_probs=True)
    
    assert len(task_types) == 1
    assert probs.shape == (1,)


def test_route_multiple_embeddings():
    output = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0]])
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    
    task_types, probs = router.route(embeddings, return_probs=True)
    
    assert len(task_types) == 3
    assert probs.shape == (3,)


def test_normalize_embeddings_numpy():
    mock_model = create_mock_model(torch.tensor([[1.0, 2.0, 3.0]]))
    router = Router(mock_model)
    
    embeddings = np.array([[0.1, 0.2, 0.3]])
    normalized = router._normalize_embeddings(embeddings)
    
    assert isinstance(normalized, torch.Tensor)
    assert torch.allclose(normalized, torch.tensor([[0.1, 0.2, 0.3]]))


def test_normalize_embeddings_torch():
    mock_model = create_mock_model(torch.tensor([[1.0, 2.0, 3.0]]))
    router = Router(mock_model)
    
    embeddings = torch.tensor([[0.1, 0.2, 0.3]])
    normalized = router._normalize_embeddings(embeddings)
    
    assert isinstance(normalized, torch.Tensor)
    assert torch.allclose(normalized, embeddings)


def test_normalize_embeddings_list():
    mock_model = create_mock_model(torch.tensor([[1.0, 2.0, 3.0]]))
    router = Router(mock_model)
    
    embeddings = [[0.1, 0.2, 0.3]]
    normalized = router._normalize_embeddings(embeddings)
    
    assert isinstance(normalized, torch.Tensor)
    assert torch.allclose(normalized, torch.tensor([[0.1, 0.2, 0.3]]))


def test_softmax_probabilities_sum_to_one():
    output = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    
    with torch.inference_mode():
        embeddings_tensor = router._normalize_embeddings(embeddings).to(router.device)
        model_output = mock_model(embeddings_tensor)
        softmax_probs = torch.nn.functional.softmax(model_output, dim=-1)
        
    row_sums = softmax_probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))


def test_predicted_class_indices_match_argmax():
    output = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    
    with torch.inference_mode():
        embeddings_tensor = router._normalize_embeddings(embeddings).to(router.device)
        model_output = mock_model(embeddings_tensor)
        softmax_probs = torch.nn.functional.softmax(model_output, dim=-1)
        _, predicted_indices = softmax_probs.max(dim=-1)
        expected_indices = model_output.argmax(dim=-1)
        
    assert torch.equal(predicted_indices, expected_indices)


def test_device_consistency():
    output = torch.tensor([[1.0, 2.0, 3.0]])
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = np.array([[0.1, 0.2, 0.3]])
    
    task_types, probs = router.route(embeddings, return_probs=True)
    
    model_device = next(mock_model.parameters()).device
    assert router.device == model_device


def test_empty_embeddings():
    output = torch.empty(0, 3)
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = np.empty((0, 10))
    
    task_types, probs = router.route(embeddings, return_probs=True)
    
    assert len(task_types) == 0
    assert probs.shape == (0,)


def test_inference_mode_context():
    output = torch.tensor([[1.0, 2.0, 3.0]])
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = np.array([[0.1, 0.2, 0.3]])
    
    with torch.enable_grad():
        task_types, probs = router.route(embeddings, return_probs=True)
        
    assert len(task_types) == 1


def test_model_called_with_correct_input():
    output = torch.tensor([[1.0, 2.0, 3.0]])
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = np.array([[0.1, 0.2, 0.3]])
    
    router.route(embeddings, return_probs=False)
    
    call_args = mock_model.call_args[0][0]
    expected_tensor = torch.tensor([[0.1, 0.2, 0.3]])
    assert torch.allclose(call_args, expected_tensor)


def test_tasktype_enum_conversion():
    output = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    
    with torch.inference_mode():
        embeddings_tensor = router._normalize_embeddings(embeddings).to(router.device)
        model_output = mock_model(embeddings_tensor)
        _, indices = torch.nn.functional.softmax(model_output, dim=-1).max(dim=-1)
        expected_indices = indices.cpu().numpy()
    
    task_types, _ = router.route(embeddings, return_probs=False)
    
    for i, task_type in enumerate(task_types):
        assert task_type == TaskType(expected_indices[i])


def test_probability_values_are_valid():
    output = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    mock_model = create_mock_model(output)
    router = Router(mock_model)
    
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    
    task_types, probs = router.route(embeddings, return_probs=True)
    
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)
    assert not torch.any(torch.isnan(probs))
    assert not torch.any(torch.isinf(probs))