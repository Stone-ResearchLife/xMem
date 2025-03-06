import time

import torch
import os
from transformers import TrainingArguments, Trainer
from perf_estimator.models import AllModels
from perf_estimator.dataset import HuggingFaceCIFAR10
from perf_estimator.config import Config
from perf_estimator.trainer.plugins import (
    ProfilerCallback,
    SnapshotCallback,
    HostMonitorCallback,
)
from perf_estimator.utilis.enum import EnumManipulator


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
torch.backends.cuda.cufft_plan_cache.max_size = 1


class ModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, loss):
        super().__init__()
        self.model = model
        self.loss = loss or torch.nn.CrossEntropyLoss()
        # Replace the final layer with a classifier for CIFAR-10.
        self.model_input_names = ["pixel_values", "labels"]

    def forward(self, pixel_values, labels=None):
        logits = self.model(pixel_values)
        if labels is not None:
            loss = self.loss(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return logits


def main(
    model: str,
    optimizer: str = "SGD",
    batch_size: int = 200,
    input_size: int = 86,
    unified_output: bool = False,
    gpu_memory_capacity: int = 4,
    cuda_enable: bool = False,
    run_id: str = None,
):
    if run_id is not None:
        config = Config(run_id=run_id, save2tmp=False)
    else:
        config = Config(save2tmp=False)

    training_args = TrainingArguments(
        output_dir=str(config.result_dir.joinpath("huggingface")),
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        logging_dir=str(config.log_dir),
        logging_steps=50,
        save_strategy="no",
        max_steps=3,
        use_cpu=(cuda_enable is False),
        do_train=True,
        # skip_memory_metrics=False, # the option may generate bulk of python_function information.
    )

    loss = torch.nn.CrossEntropyLoss()
    # _model = AllModels[model].value
    models_enum = EnumManipulator(AllModels)
    models_list = models_enum.fetch_keys()
    if model not in models_list:
        raise ValueError(
            f"Model {model} is not supported. Supported models are {','.join(models_list)}"
        )
    _model = ModelWrapper(AllModels[model].value, loss)
    _model.train()
    _dataset = HuggingFaceCIFAR10(spilt="train", image_size=(input_size, input_size))
    _optimiser = getattr(torch.optim, optimizer, torch.optim.SGD)
    _optimiser = _optimiser(params=_model.parameters(), lr=config.trainer.lr)
    _scheduler = torch.optim.lr_scheduler.StepLR(_optimiser, step_size=7, gamma=0.1)
    _callbacks = [ProfilerCallback(config=config)]
    _optimiser.zero_grad()
    _host_callback = HostMonitorCallback(
        cpu_enable=False, gpu_enable=True, network_enable=False, config=config
    )

    if cuda_enable:
        _callbacks.append(SnapshotCallback(config=config))
        _callbacks.append(_host_callback)
    _trainer = Trainer(
        model=_model,
        args=training_args,
        train_dataset=_dataset,
        callbacks=_callbacks,
        optimizers=(_optimiser, _scheduler),
    )
    try:
        _trainer.train()
    except:
        # force to stop the host monitor thread when exception occurs.
        _host_callback._monitor.stop()
        time.sleep(1)
        _host_callback._monitor.thread.join()
        real_oom = True
    else:
        real_oom = False

    if unified_output:
        import json
        import os
        from perf_estimator.utilis.utilis import filter_files
        from perf_estimator.xmem import XMem

        _output_file = config.parent_dir.joinpath("output.json")
        _results_data = {}
        if _output_file.is_file():
            with open(_output_file, "r") as f:
                _results_data = json.load(f)

        if model not in _results_data.keys():
            _results_data[model] = {}
        if optimizer not in _results_data[model].keys():
            _results_data[model][optimizer] = {}
        if str(batch_size) not in _results_data[model][optimizer].keys():
            _results_data[model][optimizer][str(batch_size)] = {}
        if "huggingface" not in _results_data[model][optimizer][str(batch_size)].keys():
            _results_data[model][optimizer][str(batch_size)]["huggingface"] = {}

        profiler_files = filter_files(
            "pt.trace.json", str(config.result_dir), fuzz=True
        )
        profiler_files.sort(key=lambda x: os.path.getmtime(x))

        data_file = profiler_files[-1]

        xmen = XMem(
            batch_size=batch_size,
            max_gpu_memory_in_gb=gpu_memory_capacity,
            config=config,
        )
        result = xmen.estimate(profiler_file=profiler_files[-1])
        _result = {"oom": real_oom, "file": data_file, "estimated": result}

        _results_data[model][optimizer][str(batch_size)]["huggingface"].update(_result)
        with open(_output_file, "w") as f:
            json.dump(_results_data, f, indent=4)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
