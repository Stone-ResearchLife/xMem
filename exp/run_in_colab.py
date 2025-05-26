import logging
import time
import torch
from exp.run import ExperimentRun, SummarySectionName
from exp.config import LargeTransformerExperiments

logging.basicConfig(level=logging.ERROR)


def main(
    model: str,
    bs: int,
    optimizer: str,
    fp16: bool = False,
):
    config = LargeTransformerExperiments()

    config.debug = False
    config.repeats = 1
    config.gpu_id = 0
    config.fp16 = fp16
    exp = ExperimentRun(config=config)
    exp.add_task(model_name=model, batch_size=bs, optimizer=optimizer, gpu_id=0)
    exp.run_group_truth()
    time.sleep(2)
    torch.cuda.empty_cache()
    time.sleep(2)
    result = exp.run_estimation(
        estimators=[SummarySectionName.solution, SummarySectionName.DNNmem]
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
