import logging
import os
import fire
from exp.trainer import FastRunner
from utils import search_profiler_file


logger = logging.getLogger(__name__)


def main(
    model: str,
    optimizer: str = "SGD",
    batch_size: int = 200,
):
    try:
        runner = FastRunner(
            model_name=model,
            batch_size=batch_size,
            optimiser=optimizer,
        )
        runner.train_on_cpu()
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise RuntimeError(f"Training failed: {e}") from e
    else:
        out_dir = runner.config.base_dir
        print(
            f"Profiled data for {model} with batch size {batch_size} and optimizer {optimizer}"
        )
        print(f"The file is saved to {out_dir}")

        c_profiling_files = search_profiler_file(out_dir)
        if len(c_profiling_files) == 0:
            raise FileNotFoundError(f"No profiling files found in {out_dir}")

        text = " Command for Estimation "
        temminal_size = os.get_terminal_size().columns
        half_temminal_size = (temminal_size - len(text)) // 2
        command_title = f"{'='*half_temminal_size}{text}{'='*half_temminal_size}"
        print(f"\033[33m{command_title}\033[0m")
        if runner.model_preparer.is_transformer:
            print(f"python main.py {c_profiling_files[-1]} -b {batch_size} -m '{model}' -i -g <int: max mem in GB>")
        else:
            print(f"python main.py {c_profiling_files[-1]} -b {batch_size} -g <int: max mem in GB>")
        print(f"\033[33m{'='*temminal_size}\033[0m")




if __name__ == "__main__":
    fire.Fire(main)
