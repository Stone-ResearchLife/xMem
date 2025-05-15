import logging
import fire
from exp.trainer import FastRunner


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
        print(
            f"Profiled data for {model} with batch size {batch_size} and optimizer {optimizer}"
        )
        print(f"The file is saved to {runner.config.result_dir}")


if __name__ == "__main__":
    fire.Fire(main)
