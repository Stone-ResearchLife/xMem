import docker
from typing import Optional, Dict, Union, List
from ures.docker.image import ImageOrchestrator, Image
from .config import Configs, BuildConfig


class ContainersReady:
    def __init__(self, user: str, client: Optional[docker.DockerClient] = None):
        self._client = client or docker.from_env()
        self._configs = Configs(username=user)
        self._image_manager = ImageOrchestrator()

    @property
    def configs(self) -> Configs:
        return self._configs

    @property
    def base_image(self) -> Dict[str, Union[Optional[Image], BuildConfig, str]]:
        image_name = "xmem-basic"
        tag = "latest"
        full_name = f"{image_name}:{tag}"
        if full_name not in self._image_manager.images.keys():
            self._add_image(
                image_name=image_name, tag=tag, config=self._configs.basic_config()
            )
        return self._image_manager.images[full_name]

    @property
    def xprofiler_image(self) -> Dict[str, Union[Optional[Image], BuildConfig, str]]:
        image_name = "xmem-xprofiler"
        tag = "latest"
        full_name = f"{image_name}:{tag}"
        if full_name not in self._image_manager.images.keys():
            self._add_image(
                image_name=image_name,
                tag=tag,
                config=self._configs.xprofiler_config(),
                base=self.base_image["image"],
            )
        return self._image_manager.images[full_name]

    @property
    def tprofiler_image(self) -> Dict[str, Union[Optional[Image], BuildConfig, str]]:
        image_name = "xmem-tprofiler"
        tag = "latest"
        full_name = f"{image_name}:{tag}"
        # Using python image inseatd of pytorch image avoids that unmatched-memory block issue occurs
        # during memory analysis of xMem
        if full_name not in self._image_manager.images.keys():
            self._add_image(
                image_name=image_name,
                tag=tag,
                config=self._configs.tprofiler_config_python_image(),
            )
        return self._image_manager.images[full_name]

    @property
    def tprofiler_image_nvidia_version(
        self,
    ) -> Dict[str, Union[Optional[Image], BuildConfig, str]]:
        image_name = "xmem-tprofiler-nvidia"
        tag = "latest"
        full_name = f"{image_name}:{tag}"
        # Using python image inseatd of pytorch image avoids that unmatched-memory block issue occurs
        # during memory analysis of xMem
        if full_name not in self._image_manager.images.keys():
            self._add_image(
                image_name=image_name,
                tag=tag,
                config=self._configs.tprofiler_config(),
                base=self.base_image["image"],
            )
        return self._image_manager.images[full_name]

    @property
    def experiment_image(self) -> Dict[str, Union[Optional[Image], BuildConfig, str]]:
        image_name = "xmem-experiments"
        tag = "latest"
        full_name = f"{image_name}:{tag}"
        if full_name not in self._image_manager.images.keys():
            self._add_image(
                image_name=image_name,
                tag=tag,
                config=self._configs.experiments_config(),
            )
        return self._image_manager.images[full_name]

    @property
    def llm_experiment_image(self) -> Dict[str, Union[Optional[Image], BuildConfig, str]]:
        image_name = "xmem-experiments-llm"
        tag = "latest"
        full_name = f"{image_name}:{tag}"
        if full_name not in self._image_manager.images.keys():
            self._add_image(
                image_name=image_name,
                tag=tag,
                config=self._configs.llm_experiments_config(),
            )
        return self._image_manager.images[full_name]

    @property
    def images(self) -> List[Dict[str, Union[Optional[Image], BuildConfig, str]]]:
        return list(self._image_manager.images.values())

    def _add_image(
        self,
        image_name: str,
        config: BuildConfig,
        base: Optional[Image] = None,
        tag: Optional[str] = None,
    ) -> Image:
        _image = Image(image_name=image_name, tag=tag, client=self._client)
        self._image_manager.add_image(image=_image, config=config, base=base)
        return _image

    def build(self):
        self._image_manager.build_all()

    def build_images(self):
        # call image functions for inserting images into the manager
        _ = self.base_image
        _ = self.xprofiler_image
        _ = self.tprofiler_image
        _ = self.tprofiler_image_nvidia_version
        _ = self.experiment_image
        _ = self.llm_experiment_image
        self._image_manager.build_all()
