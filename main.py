import fire
from typing import Union, Optional
from pathlib import Path
from perf_estimator.xmem import XMem
from perf_estimator.config import Config
from perf_estimator.log import init_logging


def main(
    profiler_file: str,
    is_transformer: bool = False,
    model_name: Optional[str] = None,
    batch_size: int = 200,
    gpu_memory_in_gb: Union[int, float] = 8,
):
    assert (
        is_transformer and model_name is not None
    ), "model_name must be provided when is_transformer is True"
    _conf = Config(save2tmp=False)
    if is_transformer:
        _conf.trainer.huggingface_enable = True
        _conf.trainer.huggingface_model_name = model_name
    init_logging(level="WARNING", conf=_conf)
    xmen = XMem(
        batch_size=batch_size,
        max_gpu_memory_in_gb=gpu_memory_in_gb,
        config=_conf,
    )
    estimated_result = xmen.estimate(profiler_file=profiler_file)
    _p_file = Path(profiler_file)
    _p_name = _p_file.name.split(".")[0]
    output_file = _p_file.parent.joinpath(f"xMem-result-{_p_name}.json")
    with open(output_file, "w") as f:
        import json

        json.dump(estimated_result, f, indent=4)
    print(f"Estimated result is saved in {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
