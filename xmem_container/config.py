import os
from pathlib import Path
from typing import Optional
from ures.docker import BuildConfig, RuntimeConfig


class Configs:
    def __init__(self, username: Optional[str] = None):
        self._username = username or "xmem"

    def user(self) -> str:
        return self._username

    def container_home_dir(self) -> Path:
        return Path(f"/home/{self.user()}")

    def host_home_dir(self) -> Path:
        return Path().home()

    def host_cache_dir(self) -> Path:
        return self.host_home_dir().joinpath(".cache", "XMemEstimator")

    def basic_config(self) -> BuildConfig:
        root_path = Path(__file__).parent.parent
        mandatory_requirements = root_path.joinpath("requirement.txt")
        _config = BuildConfig(
            base_image="pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel",
            python_deps_manager="pip",
            sys_dependencies=["vim"],
            sys_deps_manager="apt",
            labels=[
                ("Project", "xMem"),
                ("Purpose", "xMem Basic"),
            ],
            copies=[
                {
                    "src": "perf_estimator",
                    "dest": "perf_estimator",
                },
                {
                    "src": "main.py",
                    "dest": "main.py",
                },
            ],
            context_dir=Path(__file__).parent.parent,
            user=self.user(),
            uid=os.getuid(),
        )
        with open(mandatory_requirements) as f:
            # Insert the python dependencies
            for line in f.readlines():
                line = str(line).replace("\n", "").strip()
                if len(line) > 0:
                    if "[" in line and "]" in line:
                        line = "'" + line + "'"
                    _config.add_python_dependency(line)
        return _config

    def experiments_config(self) -> BuildConfig:
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
                "timm==1.0.11",
            ],
            sys_dependencies=["vim"],
            labels=[],
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
                },
            ],
            entrypoint=["python", "evaluation.py"],
            context_dir=Path(__file__).parent.parent,
        )
        return _config

    def llm_experiments_config(self) -> BuildConfig:
        _config = BuildConfig(
            base_image="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel",
            # base_image="pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel",
            python_dependencies=[
                "datasets",
                "ures",
                "transformers",
                "pynvml",
                "sortedcontainers",
                "matplotlib",
                "plotly",
                "pydantic",
                "docker",
                "fire",
                "scikit-learn",
                "numpy",
            ],
            sys_dependencies=["vim"],
            labels=[],
            copies=[
                {
                    "src": "experiments",
                    "dest": "experiments",
                },
                {
                    "src": "utility",
                    "dest": "utility",
                },
                {
                    "src": "experiments_llm",
                    "dest": "experiments_llm",
                },
                {
                    "src": "perf_estimator",
                    "dest": "perf_estimator",
                },
                {
                    "src": "experiments_llm/evaluation.py",
                    "dest": "evaluation.py",
                },
            ],
            entrypoint=["python", "evaluation.py"],
            context_dir=Path(__file__).parent.parent,
        )
        return _config

    def xprofiler_config(self) -> BuildConfig:
        _config = BuildConfig(
            labels=[
                ("Purpose", "xMem Profiler"),
            ],
            copies=[
                {
                    "src": "xProfile.py",
                    "dest": "xProfile.py",
                },
            ],
            entrypoint=["python", "xProfile.py"],
            context_dir=Path(__file__).parent.parent,
        )
        return _config

    def tprofiler_config(self) -> BuildConfig:
        _config = BuildConfig(
            labels=[
                ("Purpose", "xMem Profiler"),
            ],
            copies=[
                {
                    "src": "tProfile.py",
                    "dest": "tProfile.py",
                },
            ],
            entrypoint=["python", "tProfile.py"],
            context_dir=Path(__file__).parent.parent,
        )
        return _config

    def tprofiler_config_python_image(self) -> BuildConfig:
        root_path = Path(__file__).parent.parent
        mandatory_requirements = root_path.joinpath("requirement.txt")
        _config = BuildConfig(
            base_image="python:3.10",
            python_deps_manager="pip",
            sys_dependencies=["vim"],
            sys_deps_manager="apt",
            labels=[
                ("Project", "xMem"),
                ("Purpose", "xMem Basic"),
            ],
            python_dependencies=[
                "torchvision",
            ],
            copies=[
                {
                    "src": "perf_estimator",
                    "dest": "perf_estimator",
                },
                {
                    "src": "tProfile.py",
                    "dest": "tProfile.py",
                },
            ],
            context_dir=Path(__file__).parent.parent,
            user=self.user(),
            uid=os.getuid(),
            entrypoint=["python", "tProfile.py"],
        )
        with open(mandatory_requirements) as f:
            # Insert the python dependencies
            for line in f.readlines():
                line = str(line).replace("\n", "").strip()
                if len(line) > 0:
                    if "[" in line and "]" in line:
                        line = "'" + line + "'"
                    _config.add_python_dependency(line)
        return _config
