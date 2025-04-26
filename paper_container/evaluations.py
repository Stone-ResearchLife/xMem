import uuid
import tqdm
import time
import logging
from pathlib import Path
from typing import Optional
from ures.docker.containers import Containers
from ures.docker.runtime import SimpleRuntime
from .apps import AbcExecutor, RuntimeConfig, unique_id



class EvaluateRuntime(SimpleRuntime):
    def run(self, *args, **kwargs):
        """
        Runs each container by calling its run() method. If the container becomes running,
        wait() is called; otherwise, an error is logged.

        Returns:
            None

        Example:
            >>> runtime = SimpleRuntime([container1, container2])
            >>> runtime.run()
        """
        for container in tqdm.tqdm(self._containers):
            container.run()
            self._regular_delay()
            if container.is_running is False:
                logging.error(f"[Failed] {container.image_name} failed to start")
                continue
            else:
                timeout = 600
                start_time = time.time()
                while container.is_running:
                    time.sleep(5)
                    c_time = time.time()
                    if c_time - start_time > timeout:
                        container.stop()
                        break


class Experiments(AbcExecutor):
    def __init__(self):
        super().__init__()
        self._containers = Containers(
            image=self.image,
            client=self._client,
            runtime=EvaluateRuntime
        )
        self._schedtune_containers = Containers(
            image=self._images.schedtun_image["image"],
            client=self._client,
            runtime=EvaluateRuntime
        )
        self._llmem_containers = Containers(
            image=self._images.llmem_image["image"],
            client=self._client,
            runtime=EvaluateRuntime
        )

    @property
    def image(self):
        return self._images.paper_image['image']

    def prepare(self):
        print("Building images.")
        _ = self._images.schedtun_image
        _ = self._images.paper_image
        _ = self._images.llmem_image
        self._images.build()

    def add_container(
            self,
            model: str,
            batch: int,
            optimizer: str,
            gpu_id: int,
            task_id: Optional[str] = None,
            is_transformer: bool = True,
            dnnmem: bool = False,
            llmem: bool = False,
            schedtune: bool = False,
            paper: bool = False,
            ground: bool = False,
            debug: bool = False,
    ):
        command = [
            "--model", model.replace("/", "-"),
            "--batch", str(batch),
            "--optimizer", optimizer,
            "--batch", str(batch),
            "--optimizer", str(optimizer),
            "--gpu_id", str(gpu_id),
            "--is_transformer", is_transformer,
            "--dnnmem", dnnmem,
            "--schedtune", schedtune,
            "--paper", paper,
            "--ground", ground,
            "--debug", debug,

        ]
        if task_id is not None:
            command += ["--task_id", task_id]

        run_id = f"llm-{model.replace('/', '-')}-{optimizer}-{batch}_{uuid.uuid4().hex[:8]}"
        runtime_conf = RuntimeConfig(
            name=f"{run_id.lower()}",
            command=command,
        )
        if isinstance(gpu_id, int):
            import torch
            numer_gpu = torch.cuda.device_count()
            runtime_conf.gpus = [str(i) for i in range(numer_gpu)]

        self._add_pytorch_dataset_volume(config=runtime_conf)
        self._add_huggingface_cache_volume(config=runtime_conf)
        if is_transformer:
            dir_name = "Transformer-Exp"
        else:
            dir_name = "CNN-Exp"
        src = Path().home().joinpath(dir_name)
        dest_container = self._home_in_container().joinpath(dir_name)
        runtime_conf.add_volume(
            host_path=str(src),
            container_path=str(dest_container),
            mode="rw",
        )
        containers = []
        if schedtune:
            schedtune_conf = runtime_conf.model_copy(deep=True)
            schedtune_conf.name = f"schedtune-{runtime_conf.name}"
            containers.append(self._schedtune_containers.create(**schedtune_conf.model_dump()))

        if llmem:
            import torch
            llmem_conf = RuntimeConfig(
                name=f"llmem-{runtime_conf.name}",
                gpus=[str(gpu_id)],
            )
            llmem_conf.add_env("MODELNAME", model)
            llmem_conf.add_env("BATCH", str(batch))
            self._add_huggingface_cache_volume(config=llmem_conf)
            self._add_pytorch_dataset_volume(config=llmem_conf)
            self._add_colossalai_cache_volume(config=llmem_conf)

            task_id = f"{model.replace('/', '-')}_{optimizer}_{batch}_{gpu_id}_{task_id}"
            gpu_name = torch.cuda.get_device_properties(gpu_id).name.replace(" ", "-")

            src = Path().home().joinpath(dir_name, task_id, gpu_name)
            src.mkdir(parents=True, exist_ok=True)
            dest = self._home_in_container().joinpath("output")
            llmem_conf.add_volume(
                host_path=str(src),
                container_path=str(dest),
                mode="rw",
            )
            containers.append(self._llmem_containers.create(**llmem_conf.model_dump()))

        if paper or dnnmem or ground:
            containers.append(self._containers.create(**runtime_conf.model_dump()))

        return containers

    def _execute(self, **kwargs):
        print(
            f"=============== Start massively run for GPU train ======================"
        )
        self._containers.run()
        self._schedtune_containers.run()
        self._llmem_containers.run()

    def execute(
            self,
            model: str = "VGG16",
            batch: int = 200,
            optimizer: str = "SGD",
            gpu_id: int = 0,
            task_id: Optional[str] = None,
            is_transformer: bool = True,
            dnnmem: bool = False,
            llmem: bool = False,
            schedtune: bool = False,
            paper: bool = False,
            manual_container: bool = False,
    ):
        if manual_container is False:
            self.add_container(
                model=model,
                batch=batch,
                optimizer=optimizer,
                gpu_id=gpu_id,
                task_id=task_id,
                is_transformer=is_transformer,
                dnnmem=dnnmem,
                llmem=llmem,
                schedtune=schedtune,
                paper=paper,
            )
        self._execute()


