import os
from pathlib import Path
from ures.docker import BuildConfig, RuntimeConfig


def basic_config() -> BuildConfig:
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
        user="xmem",
        uid=os.getuid(),
    )
    with open(mandatory_requirements) as f:
        # Insert the python dependencies
        for line in f.readlines():
            line = str(line).replace("\n", "").strip()
            if len(line) > 0:
                _config.add_python_dependency(line)

    return _config


def experiments_config() -> BuildConfig:
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
            },
        ],
        entrypoint=["python", "evaluation.py"],
        context_dir=Path(__file__).parent.parent,
    )
    return _config


def xprofiler_config() -> BuildConfig:
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


def tprofiler_config() -> BuildConfig:
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
