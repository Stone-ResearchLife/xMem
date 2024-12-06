import os
import uuid
from pathlib import Path
from typing import Union, Optional
from experiments.container import AbcContainerTemplate, BuildConfig, RuntimeConfig


class SolutionEvaluation(AbcContainerTemplate):
    def __init__(
            self,
            output_dir: Optional[Union[str, os.PathLike]] = None,
            dataset_dir: Optional[Union[str, os.PathLike]] = None,
            container_name: Optional[str] = None,
    ):
        container_name = container_name or f"evaluations-{uuid.uuid4().hex[:4]}"
        super().__init__(output_dir, dataset_dir, container_name)

    @property
    def container_output_dir(self) -> Path:
        return self.container_home.joinpath("DL-Estimator")

    @property
    def username(self) -> str:
        return "evaluations"

    @property
    def context_dir(self):
        return Path(__file__).parent.parent.parent.parent

    def build_config(self, **kwargs) -> BuildConfig:
        _config = BuildConfig(
            base_image="pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel",
            python_dependencies=[
                "transformers==4.39.3",
                "pynvml==11.5.3",
                "sortedcontainers==2.4.0",
                "matplotlib~=3.9.2",
                "plotly==5.24.0",
                "pydantic==2.9.2",
                "docker==7.1.0",
                "fire~=0.7.0",
                "colossalai==0.4.5",
                "joblib==1.4.2",
                "scikit-learn==1.1.1",
                "numpy==1.26.4",
                "gputil==1.4.0",
                "timm==1.0.11"
            ],
            sys_dependencies=["vim"],
            labels=[
                ("VENUE", "ICDCS2025"),
            ],
            copies=[
                {
                    "src": "experiments",
                    "dest": "experiments",
                },
                {
                    "src": "perf_estimator",
                    "dest": "perf_estimator",
                },
                {
                    "src": "experiments/evaluation.py",
                    "dest": "evaluation.py",

                }
            ],
            entrypoint=["python", "evaluation.py"],
        )
        return _config

    def runtime_config(self, **kwargs) -> RuntimeConfig:
        _image_name = kwargs.get("image_name")
        _name = kwargs.get("name")
        _command = kwargs.get("command")
        _remove = kwargs.get("remove", False)
        _detach = kwargs.get("detach", True)
        _config = RuntimeConfig(
            image_name=_image_name,
            name=_name,
            command=_command,
            remove=_remove,
            detach=_detach,
            gpus=["0", "1"],
        )
        return _config

