from typing import List, Tuple, Optional
from ures.docker.containers import Containers
from .apps import AbcExecutor, RuntimeConfig, unique_id


class HFTComparsion(AbcExecutor):
    def __init__(self):
        super().__init__()
        self._nvidia_containers = Containers(
            image=self._images.tprofiler_image_nvidia_version["image"],
            client=self._client,
        )
        self._cpu_containers = Containers(
            image=self._images.tprofiler_image["image"],
            client=self._client,
        )

    @property
    def image(self):
        return self._images.tprofiler_image["image"]

    def _add_container(
        self,
        model: str = "VGG16",
        optimizer: str = "SGD",
        batch_size: int = 200,
        input_size: int = 86,
        unified_output: bool = False,
        gpu_memory_capacity: int = 8,
        cuda_enable: bool = False,
        run_id: Optional[str] = None,
        gpu_id: Optional[int] = None,
        memory_capacity: Optional[str] = None,
    ):
        run_id = run_id or unique_id()
        command = [
            model,
            "--optimizer",
            str(optimizer),
            "--batch_size",
            batch_size,
            "--input_size",
            input_size,
            "--unified_output",
            unified_output,
            "--gpu_memory_capacity",
            gpu_memory_capacity,
            "--cuda_enable",
            cuda_enable,
            "--run_id",
            str(run_id),
        ]
        runtime_conf = RuntimeConfig(
            name=f"{run_id}-{unique_id()[:8]}", command=command
        )
        if gpu_id is not None or cuda_enable is not None:
            if gpu_id is None:
                gpu_id = 0
            runtime_conf.gpus = [str(gpu_id)]
        if memory_capacity is not None:
            runtime_conf.memory = memory_capacity
        self._add_pytorch_dataset_volume(config=runtime_conf)
        self._add_cache_volume(config=runtime_conf)
        if cuda_enable:
            self._nvidia_containers.create(**runtime_conf.model_dump())
        else:
            self._cpu_containers.create(**runtime_conf.model_dump())

    def _execute(self, **kwargs):
        print(
            f"=============== Start massively run for GPU train ======================"
        )
        self._nvidia_containers.run()

    def execute(
        self,
        input_size: int = 86,
    ):
        models = [
            "VGG11",
            "VGG16",
            "VGG19",
            "ResNet50",
            "ResNet101",
            "ResNet152",
            "MobileNetV2",
            "MobeNetV3Small",
            "MobeNetV3Large",
            "MnasNet",
            "ConvNeXtTiny",
            "ConvNeXtBase",
            "RegNetX400MF",
            "RegNetX32GF",
            "RegNetY400MF",
            "RegNetY32GF",
        ]
        batch_size = range(10, 570, 40)
        optimizers = ["SGD", "Adam", "RMSprop", "Adagrad", "AdamW"]
        for model in models:
            for optimizer in optimizers:
                for batch in batch_size:
                    run_id = f"tprofiler-{model}-{optimizer}-batch-{batch}-Nvidia"
                    self._add_container(
                        model=model,
                        optimizer=optimizer,
                        batch_size=batch,
                        input_size=input_size,
                        unified_output=False,
                        gpu_memory_capacity=8,
                        cuda_enable=True,
                        run_id=run_id,
                        gpu_id=1,
                        memory_capacity=None,
                    )

        self._execute()
