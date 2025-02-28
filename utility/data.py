import pandas as pd
from pathlib import Path
from abc import abstractmethod, ABC
from typing import Union, List
from pydantic import BaseModel, Field, computed_field


class EstimatedMemoryRecord(BaseModel):
    tool: str = Field(
        title="Estimation Tool Name",
        description="Name of the tool used to estimate memory usage",
    )
    gt_tool: str = Field(
        title="Ground Truth Tool Name",
        description="Name of the tool used to measure ground truth memory usage",
    )
    platform: str = Field(
        title="Platform", description="Platform used for Training Job"
    )
    model: str = Field(title="Model Name", description="Name of the model")
    batch_size: int = Field(
        title="Batch Size", description="Batch size used for inference"
    )
    optimizer: str = Field(title="Optimizer", description="Optimizer used for training")
    est_memory: Union[int, float] = Field(
        title="Estimated Segment Memory Usage",
        description="Memory usage of each segment in bytes",
    )
    gt_memory: Union[int, float] = Field(
        title="Ground Truth Segment Memory Usage",
        description="Ground truth memory usage in bytes",
    )
    runtime: Union[int, float] = Field(
        title="Running time of each estimation", description="Runtime in seconds"
    )
    gpu_capacity: Union[int, float] = Field(
        title="GPU Capacity", description="GPU capacity in bytes"
    )
    est_oom: bool = Field(
        title="Estimated Out of Memory",
        description="Whether the model runs out of memory",
    )
    oom: bool = Field(
        title="Real Out of Memory", description="Whether the model runs out of memory"
    )
    accuracy_mode: bool = Field(
        title="Accuracy Mode",
        description="Whether the gpu capacity is set to the maximum memory usage estimated by tool",
    )

    @computed_field
    @property
    def relative_error(self) -> float:
        """
        Calculate the relative error of the estimated memory usage in precentage.
        Maximum error is 100%.
        """
        if self.gt_memory == 0:
            return 100
        else:
            return abs(
                round(((self.est_memory - self.gt_memory) / self.gt_memory) * 100, 2)
            )


class DataProcessorInterface(ABC):
    @abstractmethod
    def get_data(self, *args, **kwargs) -> List[EstimatedMemoryRecord]:
        pass


class DataAggregation:
    def __init__(self):
        self._estimated_results: List[EstimatedMemoryRecord] = []

    @property
    def results(self) -> List[EstimatedMemoryRecord]:
        return self._estimated_results

    @results.setter
    def results(self, results: List[EstimatedMemoryRecord]):
        assert isinstance(results, list)
        assert (
            all([isinstance(data, EstimatedMemoryRecord) for data in results]) is True
        )
        self._estimated_results = results

    def load(self, josn_file: str):
        import json

        with open(josn_file, "r") as f:
            data = json.load(f)
        self.results = [EstimatedMemoryRecord(**record) for record in data]

    def process(self, processor: DataProcessorInterface):
        self.results = processor.get_data()

    def to_dict(self) -> List[dict]:
        return [data.model_dump() for data in self.results]

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict())

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=4)

    def save(self, dest: Union[Path, str], format: str = "json"):
        if format.lower() == "csv":
            self.to_df().to_csv(dest, index=True)
        else:
            with open(dest, "w") as f:
                f.write(self.to_json())
