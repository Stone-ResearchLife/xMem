import docker
import logging
from pathlib import Path
from typing import Optional

from traitlets import directional_link
from ures.docker import RuntimeConfig, Container
from ures.string import unique_id

from xmem_container.containers import ContainersReady

logger = logging.getLogger(__name__)


class XMem:
    def __init__(self):
        self._client = docker.from_env()
        self._images = ContainersReady(client=self._client)

    @property
    def user_name(self):
        return "xmem"

    def _image_ready_check(self) -> bool:
        pass_flag = True
        for image_dict in self._images.images:
            _image = image_dict['image']
            if _image.exist is False:
                pass_flag = False
        return pass_flag

    def prepare(self):
        if self._image_ready_check() is False:
            logger.warning(f"As some images are not ready, we will build them now.")
            self._images.build_images()

    def _add_pytorch_dataset_volume(self, config: RuntimeConfig):
        dir_in_host = Path().home().joinpath("pytorch_datasets")
        dir_in_host.mkdir(exist_ok=True, parents=True)
        dir_in_container = Path(f"/home/{self._images._configs.user()}/pytorch_datasets")
        config.add_volume(
            host_path=str(dir_in_host),
            container_path=str(dir_in_container),
            mode="rw"
        )

    def _add_cache_volume(self, config: RuntimeConfig):
        dir_in_host = Path().home().joinpath(".cache", "XMemEstimator")
        dir_in_host.mkdir(exist_ok=True, parents=True)
        dir_in_container = Path(f"/home/{self._images._configs.user()}/DL-Estimator")
        config.add_volume(
            host_path=str(dir_in_host),
            container_path=str(dir_in_container),
            mode="rw"
        )


    def x_profiler(
            self,
            model: str = "VGG16",
            optimizer: str = "SGD",
            batch_size: int = 200,
            input_size: int = 86,
            unified_output: bool = False,
            gpu_memory_capcity: int = 8,
            cuda_enabled: bool = False,
            run_id: Optional[str] = None,
    ):
        run_id = run_id or unique_id()
        command = [
            model,
            "--optimizer", str(optimizer),
            "--batch_size", batch_size,
            "--input_size", input_size,
            "--unified_output", unified_output,
            "--gpu_memory_capacity", gpu_memory_capcity,
            "--cuda_enabled", cuda_enabled,
            "--run_id", str(run_id)
        ]
        container = Container(
            image=self._images.xprofiler_image['image'],
            client=self._client
        )
        runtime_conf = RuntimeConfig(
            name=f"xprofiler-{run_id}",
            command=command
        )
        self._add_pytorch_dataset_volume(config=runtime_conf)
        self._add_cache_volume(config=runtime_conf)
        container.create(config=runtime_conf)
        container.run()
        if container.is_running:
            container.wait()


if __name__ == '__main__':
    xmem = XMem()
    xmem.x_profiler()