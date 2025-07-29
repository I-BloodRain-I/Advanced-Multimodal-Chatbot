from typing import Any

from diffusers import (
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler
)

def get_scheduler(pipeline, scheduler_type="euler") -> Any:
    """
    Returns a diffusion scheduler instance configured from a pipeline.

    Args:
        pipeline: The diffusion pipeline to configure the scheduler for.
        scheduler_type: Type of scheduler to use. One of:
            ['euler', 'euler_ancestral', 'dpm', 'ddim', 'lms'].

    Returns:
        An instance of the selected scheduler type.
    """
    schedulers = {
        "euler": EulerDiscreteScheduler,                 # Fast, good quality (20–30 steps)
        "euler_ancestral": EulerAncestralDiscreteScheduler,  # Slightly more diverse (20–40 steps)
        "dpm": DPMSolverMultistepScheduler,              # Very fast (20–25 steps)
        "ddim": DDIMScheduler,                           # Slower but stable (50–100 steps)
        "lms": LMSDiscreteScheduler                      # Legacy alternative (30–50 steps)
    }
    if scheduler_type not in schedulers:
        raise ValueError(f"Invalid scheduler_type '{scheduler_type}'. Valid options are: {list(schedulers.keys())}")

    return schedulers[scheduler_type].from_config(pipeline.scheduler.config)