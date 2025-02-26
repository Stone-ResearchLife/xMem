import docker
from typing import Optional
from ures.docker.image import ImageOrchestrator, Image
from containers.config import *


class xMemImages:
    def __init__(self, client: Optional[docker.DockerClient] = None):
        self._manager = ImageOrchestrator()
        self._client = client or docker.from_env()

    def _add_image(
        self, image_name: str, config: BuildConfig, base: Optional[Image] = None
    ) -> Image:
        _image = Image(image_name=image_name, client=self._client)
        self._manager.add_image(image=_image, config=config, base=base)
        return _image

    def prepare(self):
        base_image = self._add_image(image_name="xmem-basic", config=basic_config())
        exp_image = self._add_image(
            image_name="xmem-experiments",
            config=experiments_config(),
        )
        xprofiler_image = self._add_image(
            image_name="xmem-xprofiler", config=xprofiler_config(), base=base_image
        )
        tprofiler_image = self._add_image(
            image_name="xmem-tprofiler", config=tprofiler_config(), base=base_image
        )

    def build(self):
        self._manager.build_all()
