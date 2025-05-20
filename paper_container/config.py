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

    # for experiments of recent version
    def exp_base_config(self) -> BuildConfig:
        _config = BuildConfig(
            base_image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
            python_dependencies=[
                "transformers==4.51.3",
                "pynvml==12.0.0",
                "sortedcontainers==2.4.0",
                "matplotlib~=3.10.1",
                "plotly==6.0.0",
                "pydantic==2.9.2",
                "tqdm==4.67.1",
                "docker==7.1.0",
                "fire~=0.7.0",
                "datasets==3.5.0",
                "gputil==1.4.0",
                "pandas==2.2.3",
            ],
            sys_dependencies=["vim"],
            labels=[],
            context_dir=Path(__file__).parent.parent,
            user=self.user(),
            uid=os.getuid(),
        )
        return _config

    def paper_config(self) -> BuildConfig:
        _config = BuildConfig(
            python_dependencies=[
                "ures==1.9.0",
            ],
            labels=[
                ("Platform", "PyTorch"),
            ],
            copies=[
                {
                    "src": "exp",
                    "dest": "exp",
                },
                {
                    "src": "utils",
                    "dest": "utils",
                },
                {
                    "src": "perf_estimator",
                    "dest": "perf_estimator",
                },
                {
                    "src": "paper_container",
                    "dest": "paper_container",
                },
                {
                    "src": "exp/run.py",
                    "dest": "run.py",
                },
            ],
            entrypoint=["python", "run.py"],
            context_dir=Path(__file__).parent.parent,
        )
        return _config

    def schedtune_config(self) -> BuildConfig:
        _config = BuildConfig(
            python_dependencies=[
                "ures==1.9.0",
                "pandas==2.2.3",
                "datasets==3.5.0",
                "torchinfo==1.8.0",
            ],
            labels=[
                ("Platform", "PyTorch"),
            ],
            copies=[
                {
                    "src": "exp",
                    "dest": "exp",
                },
                {
                    "src": "utils",
                    "dest": "utils",
                },
                {
                    "src": "perf_estimator",
                    "dest": "perf_estimator",
                },
                {
                    "src": "paper_container",
                    "dest": "paper_container",
                },
                {
                    "src": "exp/run.py",
                    "dest": "run.py",
                },
            ],
            entrypoint=["python", "run.py"],
            context_dir=Path(__file__).parent.parent,
        )
        return _config

    def llmem_config(self) -> BuildConfig:
        _config = BuildConfig(
            base_image="llmem:latest",
            python_dependencies=[
                "ures==1.9.0",
                "pandas==2.2.3",
            ],
            sys_dependencies=["vim"],
            copies=[
                {
                    "src": "exp/baselines/LLmem",
                    "dest": ".",
                },
            ],
            context_dir=Path(__file__).parent.parent,
            user=self.user(),
            uid=os.getuid(),
            entrypoint=["bash", "run_colo.sh"],
        )
        return _config

    # for previous version of experiments
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
                # {
                #     "src": "experiments",
                #     "dest": "experiments",
                # },
                {
                    "src": "perf_estimator",
                    "dest": "perf_estimator",
                },
                # {
                    # "src": "experiments/evaluation.py",
                    # "dest": "evaluation.py",
                # },
            ],
            entrypoint=["python", "evaluation.py"],
            context_dir=Path(__file__).parent.parent,
            user=self.user(),
            uid=os.getuid(),
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
