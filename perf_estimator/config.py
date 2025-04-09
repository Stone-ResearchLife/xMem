import uuid
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from perf_estimator.utilis.utilis import temp_dir_with_specific_path, time_now


class DatasetConfig(BaseModel):
    shuffle: bool = Field(default=True, description="Shuffle Dataset")


class TrainerConfig(BaseModel):
    lr: float = Field(default=0.001, description="Learning Rate")
    epochs: int = Field(default=1, description="Number of Epochs")
    huggingface_enable: bool = Field(default=True, description="HuggingFace Enable")
    huggingface_model_name: str = Field(default=None, description="HuggingFace Model")


class Config(BaseModel):
    model_config = ConfigDict()

    name: str = Field(default="DL-Estimator", description="Project Name")
    debug: bool = Field(default=False, description="Debug Mode")
    save2tmp: bool = Field(default=True, description="Save to Temp Directory")
    run_id: str = Field(
        default=f"{time_now(iso8601=False)}-{str(uuid.uuid4().hex)[:4]}",
        description="Run ID",
    )
    task_id: Optional[str] = Field(default=None, description="This is a subtask id, that is used to identify the task in the same run_id")
    dataset: DatasetConfig = DatasetConfig()
    trainer: TrainerConfig = TrainerConfig()

    @property
    def parent_dir(self) -> Path:
        return self.base_dir.parent

    @property
    def base_dir(self):
        if self.save2tmp:
            _base_dir = Path(temp_dir_with_specific_path(self.name, self.run_id))
        else:
            if self.task_id is None:
                _base_dir = Path().home().joinpath(self.name, self.run_id)
            else:
                _base_dir = Path().home().joinpath(self.name, self.run_id, self.task_id)
        _base_dir.mkdir(parents=True, exist_ok=True)
        return _base_dir

    @property
    def log_dir(self) -> Path:
        _path = self.base_dir.joinpath("logs")
        _path.mkdir(parents=True, exist_ok=True)
        return _path

    @property
    def result_dir(self) -> Path:
        _path = self.base_dir.joinpath("results")
        _path.mkdir(parents=True, exist_ok=True)
        return _path

    @property
    def dataset_dir(self) -> Path:
        _path = Path().home().joinpath("pytorch_datasets")
        _path.mkdir(parents=True, exist_ok=True)
        return _path

    @property
    def permanent_gpu_memory_in_gb(self) -> dict[int, float]:
        return {
            0: round(float(280 / 1024), 2),
            1: round(float(15 / 1024), 2),
        }


default_setting = Config(save2tmp=False)
