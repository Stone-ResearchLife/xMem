from pydantic import BaseModel, Field
from perf_estimator.models import AllModels
from ures.tools.enum import EnumManipulator


class ExperimentConfig(BaseModel):
    """Configuration for the experiment."""
    batch_range: tuple[int] = Field(default=(100, 200), description="Batch size for the experiment.")
    optimisers: list[str] = Field(default=["AdamW"], description="Optimiser to use for the experiment.")
    run_id: str = Field(default="default_run", description="Run ID for the experiment.")
    models: list[str] = Field(default=[], description="Models to use.")
    repeats: int = Field(default=1, description="Number of times to run the experiment.")


class CNNExperiments(ExperimentConfig):
    """Configuration for CNN experiments."""
    batch_range: tuple[int] = Field(default=(50, 500, 50), description="Batch size for the experiment.")
    run_id: str = Field(default="CNN-Exp", description="Run ID for the experiment.")
    optimisers: list[str] = Field(
        default=["SGD", "Adam", "RMSprop", "Adagrad", "AdamW"],
        description="Optimiser to use for the experiment."
    )
    models: list[str] = Field(default=EnumManipulator(AllModels).fetch_keys(), description="List of CNN models.")


class TransformerExperiments(ExperimentConfig):
    """Configuration for transformer experiments."""
    batch_range: tuple[int] = Field(default=(5, 70, 5), description="Batch size for the experiment.")
    run_id: str = Field(default="Transformer-Exp", description="Run ID for the experiment.")
    models: list[str] = Field(
        default=[
            "EleutherAI/gpt-Vneo-125M",
            "facebook/opt-125m",
            "facebook/opt-350m",
            "cerebras/Cerebras-GPT-111M",
            "microsoft/deberta-base",
            "T5-small",
            "t5-base",
            "distilbert/distilgpt2",
            "openai-community/gpt2",
        ],
        description="List of transformer models.",
    )
