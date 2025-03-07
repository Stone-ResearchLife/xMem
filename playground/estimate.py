from ures.files import filter_files
from perf_estimator.xmem import XMem
from pathlib import Path


if __name__ == "__main__":

    profiler_files = filter_files(
        "pt.trace.json", str(Path().cwd().joinpath("Profiler")), fuzz=True
    )[-1]

    xmen = XMem(
        batch_size=1,
        max_gpu_memory_in_gb=8,
    )
    result = xmen.estimate(
        profiler_file=profiler_files, trainer_enable=True, output_only=True
    )
    print(f"Profiler file: {profiler_files}")
    print(result)
