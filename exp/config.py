from pydantic import BaseModel, Field
from perf_estimator.models import AllModels
from ures.tools.enum import EnumManipulator


class ExperimentConfig(BaseModel):
    """Configuration for the experiment."""

    batch_range: tuple[int] = Field(
        default=(100, 200), description="Batch size for the experiment."
    )
    optimisers: list[str] = Field(
        default=["AdamW"], description="Optimiser to use for the experiment."
    )
    run_id: str = Field(default="default_run", description="Run ID for the experiment.")
    models: list[str] = Field(default=[], description="Models to use.")
    repeats: int = Field(
        default=1, description="Number of times to run the experiment."
    )
    gpu_id: int = Field(default=0, description="GPU ID to use.")
    debug: bool = Field(default=False, description="Enable debug mode.")
    fp16: bool = Field(default=False, description="Enable FP16 mode.")


class CNNExperiments(ExperimentConfig):
    """Configuration for CNN experiments."""

    batch_range: tuple[int] = Field(
        default=(200, 800, 100), description="Batch size for the experiment."
    )
    run_id: str = Field(default="CNN-Exp", description="Run ID for the experiment.")
    optimisers: list[str] = Field(
        default=["SGD", "Adam", "RMSprop", "Adagrad", "AdamW"],
        description="Optimiser to use for the experiment.",
    )
    models: list[str] = Field(
        default=EnumManipulator(AllModels).fetch_keys(),
        description="List of CNN models.",
    )


class TransformerExperiments(ExperimentConfig):
    """Configuration for transformer experiments."""

    batch_range: tuple[int] = Field(
        default=(5, 60, 5), description="Batch size for the experiment."
    )
    run_id: str = Field(
        default="Transformer-Exp", description="Run ID for the experiment."
    )
    models: list[str] = Field(
        default=[
            "EleutherAI/gpt-neo-125M",
            "facebook/opt-125m",
            "facebook/opt-350m",
            "cerebras/Cerebras-GPT-111M",
            "T5-small",
            "t5-base",
            "distilbert/distilgpt2",
            "openai-community/gpt2",
        ],
        description="List of transformer models.",
    )
    optimisers: list[str] = Field(
        default=["SGD", "Adam", "AdamW", "Adafactor"],
        description="Optimiser to use for the experiment.",
    )

class LargeTransformerExperiments(TransformerExperiments):
    """Configuration for transformer experiments."""
    batch_range: tuple[int] = Field(
        default=(1, 9, 2), description="Batch size for the experiment."
    )
    run_id: str = Field(
        default="Large-Transformer-Exp", description="Run ID for the experiment."
    )
    models: list[str] = Field(
        default=[
            "EleutherAI/pythia-1b",
            "Qwen/Qwen3-0.6B",
        ],
        description="List of transformer models.",
    )
    optimisers: list[str] = Field(
        default=[
            "Adafactor",
            "AdamW",
            "Adam",
            "SGD"
        ],
        description="Optimiser to use for the experiment.",
    )
