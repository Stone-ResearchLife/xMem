import fire
from typing import Union
from perf_estimator.xmem import XMem
from perf_estimator.config import Config
from perf_estimator.log import init_logging


def main(
    profiler_file: str,
    batch_size: int = 200,
    input_size: int = 86,
    gpu_memory_in_gb: Union[int, float] = 4,
):
    _conf = Config(save2tmp=False)
    init_logging(level="WARNING", conf=_conf)
    xmen = XMem(
        batch_size=batch_size,
        max_gpu_memory_in_gb=gpu_memory_in_gb,
        input_size=input_size,
        config=_conf,
    )

    estimated_result = xmen.estimate(profiler_file=profiler_file)
    data_dir = xmen.conf.result_dir
    output_file = data_dir.joinpath("estimated_result.json")
    with open(output_file, "w") as f:
        import json

        json.dump(estimated_result, f, indent=4)
    print(f"Estimated result is saved in {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
