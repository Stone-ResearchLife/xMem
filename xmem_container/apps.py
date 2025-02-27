import docker
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union
from ures.docker import RuntimeConfig, Container
from ures.string import unique_id
from xmem_container.containers import ContainersReady, Image

logger = logging.getLogger(__name__)


class AbcExecutor(ABC):
    def __init__(self):
        self._client = docker.from_env()
        self._images = ContainersReady(user=self.user_name, client=self._client)

    @property
    def user_name(self):
        return "xmem"

    @property
    @abstractmethod
    def image(self) -> Image:
        pass

    def _home_in_container(self) -> Path:
        return Path(f"/home/{self._images._configs.user()}")

    def prepare(self):
        print("Building all images.")
        self._images.build_images()
        _check = all([image_dict["image"].exist for image_dict in self._images.images])
        if _check is False:
            print("All images are ready.")
        else:
            print("Some images build failed.")

    def _add_pytorch_dataset_volume(self, config: RuntimeConfig):
        dir_in_host = Path().home().joinpath("pytorch_datasets")
        dir_in_host.mkdir(exist_ok=True, parents=True)
        dir_in_container = self._home_in_container().joinpath("pytorch_datasets")
        config.add_volume(
            host_path=str(dir_in_host), container_path=str(dir_in_container), mode="rw"
        )

    def _add_cache_volume(self, config: RuntimeConfig):
        dir_in_host = Path().home().joinpath(".cache", "XMemEstimator")
        dir_in_host.mkdir(exist_ok=True, parents=True)
        dir_in_container = self._home_in_container().joinpath("DL-Estimator")
        config.add_volume(
            host_path=str(dir_in_host), container_path=str(dir_in_container), mode="rw"
        )

    def _execute(self, **kwargs):
        runtime_config: RuntimeConfig = kwargs.pop("config")
        print(
            f"================ Container {self.image.name} is initializing ================"
        )
        container = Container(image=self.image, client=self._client)
        print(
            f"================ Container {self.image.name} is created ================"
        )
        print(f"Config: {runtime_config.model_dump()}")
        container.create(config=runtime_config)
        print(
            f"================ Container {self.image.name} is started ================"
        )
        container.run()
        if container.is_running:
            print(
                f"================ Container {self.image.name} is running ==============="
            )
            container.wait()

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass


class XProfiler(AbcExecutor):
    @property
    def image(self) -> Image:
        return self._images.xprofiler_image["image"]

    def execute(
        self,
        model: str = "VGG16",
        optimizer: str = "SGD",
        batch_size: int = 200,
        input_size: int = 86,
        unified_output: bool = False,
        gpu_memory_capcity: int = 8,
        cuda_enabled: bool = False,
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
            gpu_memory_capcity,
            "--cuda_enabled",
            cuda_enabled,
            "--run_id",
            str(run_id),
        ]
        runtime_conf = RuntimeConfig(name=f"xprofiler-{run_id}", command=command)
        if gpu_id is not None:
            runtime_conf.gpus = [str(gpu_id)]
        if memory_capacity is not None:
            runtime_conf.memory = memory_capacity
        self._add_pytorch_dataset_volume(config=runtime_conf)
        self._add_cache_volume(config=runtime_conf)
        self._execute(config=runtime_conf)


class TProfiler(AbcExecutor):
    @property
    def image(self) -> Image:
        return self._images.tprofiler_image["image"]

    def execute(
        self,
        model: str = "VGG16",
        optimizer: str = "SGD",
        batch_size: int = 200,
        input_size: int = 86,
        unified_output: bool = False,
        gpu_memory_capcity: int = 8,
        cuda_enabled: bool = False,
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
            gpu_memory_capcity,
            "--cuda_enabled",
            cuda_enabled,
            "--run_id",
            str(run_id),
        ]
        runtime_conf = RuntimeConfig(name=f"tprofiler-{run_id}", command=command)
        if gpu_id is not None:
            runtime_conf.gpus = [str(gpu_id)]
        if memory_capacity is not None:
            runtime_conf.memory = memory_capacity
        self._add_pytorch_dataset_volume(config=runtime_conf)
        self._add_cache_volume(config=runtime_conf)
        self._execute(config=runtime_conf)


class XMem(AbcExecutor):
    @property
    def image(self) -> Image:
        return self._images.base_image["image"]

    def _cache_dir(self):
        return Path().home().joinpath(".cache", "profiler_files")

    def _mount_profiler_dir(self, config: RuntimeConfig, profiler_file: str) -> Path:
        profiler_file = Path(profiler_file)
        profiler_dir = profiler_file.parent
        profiler_name = profiler_file.name
        config.add_volume(
            host_path=str(profiler_dir),
            container_path=str(self._cache_dir()),
            mode="ro",
        )
        return self._cache_dir().joinpath(profiler_name)

    def execute(
        self,
        profiler_file: str,
        batch_size: int = 200,
        input_size: int = 86,
        gpu_memory_in_gb: Union[int, float] = 4,
    ):
        runtime_conf = RuntimeConfig(
            name=f"xmem-{unique_id()[0:8]}",
        )

        profiler_path_in_container = self._mount_profiler_dir(
            config=runtime_conf, profiler_file=profiler_file
        )
        entrypoint = ["python", "main.py"]
        command = [
            "--profiler_file",
            str(profiler_path_in_container),
            "--batch_size",
            batch_size,
            "--input_size",
            input_size,
            "--gpu_memory_in_gb",
            gpu_memory_in_gb,
        ]
        runtime_conf.command = command
        runtime_conf.entrypoint = entrypoint
        self._add_pytorch_dataset_volume(config=runtime_conf)
        self._add_cache_volume(config=runtime_conf)
        self._execute(config=runtime_conf)
