import pytest
import torch
from unittest.mock import Mock, patch

from modules.image_generation.image_generator import ImageGenerator

TEST_MODEL_NAME = "test-model"
TEST_DEVICE = "cpu"
TEST_DTYPE = "fp32"
TEST_SCHEDULER_TYPE = "euler_ancestral"
TEST_WIDTH = 512
TEST_HEIGHT = 512
TEST_GUIDANCE_SCALE = 3.0
TEST_INFERENCE_STEPS = 30


@pytest.fixture(autouse=True)
def reset_singleton():
   ImageGenerator._instance = None
   yield
   ImageGenerator._instance = None


@pytest.fixture
def mock_prompts():
   return ["test prompt 1", "test prompt 2"]


@pytest.fixture
def mock_negative_prompts():
   return ["bad quality", "blurry"]


@pytest.fixture
def mock_pipeline():
   pipeline = Mock()
   pipeline.return_value = ''
   pipeline.scheduler = Mock()
   pipeline.to.return_value = pipeline
   pipeline.set_progress_bar_config = Mock()
   return pipeline


@pytest.fixture
def mock_output():
   output = Mock()
   output.images = [torch.randn(3, 512, 512), torch.randn(3, 512, 512)]
   return output


def test_singleton_pattern():
   with patch.object(ImageGenerator, '_load_pipeline'), \
        patch('shared.config.Config.get', return_value='models'), \
        patch('shared.utils.get_torch_device', return_value=torch.device(TEST_DEVICE)):
       
       gen1 = ImageGenerator(TEST_MODEL_NAME)
       gen2 = ImageGenerator("another-model")
       
       assert gen1 is gen2
       assert id(gen1) == id(gen2)


@pytest.mark.parametrize("model_exists, expect_save", [
    (True, False),
    (False, True)
])
def test_load_pipeline_model_path_behavior(mock_pipeline, model_exists, expect_save):
    with patch('shared.config.Config.get', return_value='models'), \
         patch('shared.utils.get_torch_device', return_value=torch.device(TEST_DEVICE)), \
         patch('pathlib.Path.exists', return_value=model_exists), \
         patch('diffusers.EulerAncestralDiscreteScheduler.from_config', return_value={}), \
         patch('diffusers.AutoPipelineForText2Image.from_pretrained') as mock_class:

        mock_class.return_value = mock_pipeline

        generator = ImageGenerator(TEST_MODEL_NAME, 
                                   device=TEST_DEVICE,
                                   dtype=TEST_DTYPE,
                                   scheduler_type=TEST_SCHEDULER_TYPE)

        mock_class.assert_called_once()
        if expect_save:
            mock_pipeline.save_pretrained.assert_called_once()
        else:
            mock_pipeline.save_pretrained.assert_not_called()


def test_generate_basic(mock_prompts, mock_output):
   with patch.object(ImageGenerator, '_load_pipeline') as mock_load, \
        patch('shared.config.Config.get', return_value='models'), \
        patch('shared.utils.get_torch_device', return_value=torch.device(TEST_DEVICE)):
       
       mock_pipeline = Mock()
       mock_pipeline.return_value = mock_output
       mock_load.return_value = mock_pipeline
       
       generator = ImageGenerator(TEST_MODEL_NAME)
       
       result = generator.generate(mock_prompts)
       
       mock_pipeline.assert_called_once()
       assert result == mock_output.images


import pytest

@pytest.mark.parametrize("negative_prompts, expect_default", [
    (["bad quality", "blurry"], False),
    (None, True)
])
def test_generate_negative_prompt_handling(mock_prompts, mock_output, negative_prompts, expect_default):
    with patch.object(ImageGenerator, '_load_pipeline') as mock_load, \
         patch('shared.config.Config.get', return_value='models'), \
         patch('shared.utils.get_torch_device', return_value=torch.device(TEST_DEVICE)):

        mock_pipeline = Mock()
        mock_pipeline.return_value = mock_output
        mock_load.return_value = mock_pipeline

        generator = ImageGenerator(TEST_MODEL_NAME)
        result = generator.generate(mock_prompts, negative_prompts)

        args, kwargs = mock_pipeline.call_args
        if expect_default:
            assert len(kwargs['negative_prompt']) == len(mock_prompts)
            assert all("deformed" in neg for neg in kwargs['negative_prompt'])
        else:
            assert kwargs['negative_prompt'] == negative_prompts


@pytest.mark.parametrize("use_refiner, expected_calls", [
    (True, 2),
    (False, 1)
])
def test_initialization_refiner_option(use_refiner, expected_calls):
    with patch.object(ImageGenerator, '_load_pipeline') as mock_load, \
         patch('shared.config.Config.get', return_value='models'), \
         patch('shared.utils.get_torch_device', return_value=torch.device(TEST_DEVICE)):

        mock_load.return_value = Mock()

        generator = ImageGenerator(
            model_name=TEST_MODEL_NAME,
            use_refiner=use_refiner,
            refiner_name="test-refiner" if use_refiner else None
        )

        assert mock_load.call_count == expected_calls
        assert (generator.refiner is not None) == use_refiner


def test_generate_prompt_negative_prompt_length_mismatch():
   with patch.object(ImageGenerator, '_load_pipeline'), \
        patch('shared.config.Config.get', return_value='models'), \
        patch('shared.utils.get_torch_device', return_value=torch.device(TEST_DEVICE)):
       
       generator = ImageGenerator(TEST_MODEL_NAME)
       
       with pytest.raises(ValueError, match="Mismatch between prompt and negative prompt lengths"):
           generator.generate(["prompt1", "prompt2"], ["negative1"])


def test_generate_exception_handling(mock_prompts):
   with patch.object(ImageGenerator, '_load_pipeline') as mock_load, \
        patch('shared.config.Config.get', return_value='models'), \
        patch('shared.utils.get_torch_device', return_value=torch.device(TEST_DEVICE)):
       
       mock_pipeline = Mock()
       mock_pipeline.side_effect = Exception("Pipeline error")
       mock_load.return_value = mock_pipeline
       
       generator = ImageGenerator(TEST_MODEL_NAME)
       
       with pytest.raises(Exception):
           generator.generate(mock_prompts)


def test_dtype_configuration():
   with patch.object(ImageGenerator, '_load_pipeline') as mock_load, \
        patch('shared.config.Config.get', return_value='models'), \
        patch('shared.utils.get_torch_device', return_value=torch.device(TEST_DEVICE)):
       
       mock_load.return_value = Mock()
       
       generator_fp16 = ImageGenerator(TEST_MODEL_NAME, dtype="fp16")
       assert generator_fp16._torch_dtype == torch.half
       
       ImageGenerator._instance = None
       
       generator_fp32 = ImageGenerator(TEST_MODEL_NAME, dtype="fp32")
       assert generator_fp32._torch_dtype == torch.float


def test_generate_custom_parameters(mock_prompts, mock_output):
   with patch.object(ImageGenerator, '_load_pipeline') as mock_load, \
        patch('shared.config.Config.get', return_value='models'), \
        patch('shared.utils.get_torch_device', return_value=torch.device(TEST_DEVICE)):
       
       mock_pipeline = Mock()
       mock_pipeline.return_value = mock_output
       mock_load.return_value = mock_pipeline
       
       generator = ImageGenerator(TEST_MODEL_NAME)
       
       custom_width = 1024
       custom_height = 768
       custom_guidance = 7.5
       custom_steps = 50
       
       result = generator.generate(
           mock_prompts,
           width=custom_width,
           height=custom_height,
           guidance_scale=custom_guidance,
           num_inference_steps=custom_steps
       )
       
       mock_pipeline.assert_called_once()
       args, kwargs = mock_pipeline.call_args
       assert kwargs['width'] == custom_width
       assert kwargs['height'] == custom_height
       assert kwargs['guidance_scale'] == custom_guidance
       assert kwargs['num_inference_steps'] == custom_steps