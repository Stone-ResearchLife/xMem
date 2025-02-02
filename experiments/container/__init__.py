import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Union
from pathlib import Path
from perf_estimator.utilis.utilis import fetch_temp_folder
from .image import BuildConfig, ContainerImage, ImageInfo
from .container import RuntimeConfig, ContainerRunner


logger = logging.getLogger(__name__)


class AbcContainerTemplate(ABC):
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        dataset_dir: Optional[Union[str, Path]] = None,
        container_name: Optional[str] = None,
    ):
        _output_dir = output_dir or fetch_temp_folder()
        self._output_dir = Path(_output_dir)
        self._dataset_dir = dataset_dir
        self._name = container_name

    @property
    @abstractmethod
    def container_output_dir(self) -> Path:
        pass

    @property
    @abstractmethod
    def username(self) -> str:
        pass

    @property
    @abstractmethod
    def context_dir(self) -> Path:
        pass

    @abstractmethod
    def build_config(self, **kwargs) -> BuildConfig:
        pass

    @abstractmethod
    def runtime_config(self, **kwargs) -> RuntimeConfig:
        pass

    # ============= End Abstract Properties =============

    @property
    def container_home(self) -> Path:
        return Path(f"/home/{self.username}")

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def build_image(self, force: bool = False, **kwargs) -> ImageInfo:
        """Build the container image

        Args:
            force (bool): Force rebuild the container
            **kwargs: Additional arguments for build configuration
        """
        build_dir = self._output_dir.joinpath("build")
        build_dir.mkdir(parents=True, exist_ok=True)
        build_config = self.build_config(**kwargs)
        if str(build_config.context_dir) != str(self.context_dir):
            logger.warning(
                f"Context dir: {build_config.context_dir} != {self.context_dir}"
            )
            logger.warning(f"Context dir is updated to {self.context_dir}")
            build_config.context_dir = self.context_dir
        if build_config.user is None or build_config.user != self.username:
            logger.warning(f"User: {build_config.user} != {self.username}")
            logger.warning(
                f"User and Id are updated to {self.username} and {os.getuid()}"
            )
            build_config.user = self.username
            build_config.uid = os.getuid()
        _image = ContainerImage(
            image_name=self._name,
            config=build_config,
            rebuild=force,
            temp_dir=build_dir,
        )
        return _image.get_image()

    def run_image(self, **kwargs) -> ContainerRunner:
        _run_conf = self.runtime_config(**kwargs)
        _run_conf.build_config = self.build_config()
        if self.output_dir is not None:
            _run_conf.add_volume(
                host_path=self.output_dir,
                container_path=self.container_output_dir,
                mode="rw",
            )
        if self._dataset_dir is not None:
            _run_conf.add_volume(
                host_path=self._dataset_dir,
                container_path=self.container_home.joinpath(self._dataset_dir.name),
                mode="rw",
            )
        _runner = ContainerRunner(config=_run_conf)
        _runner.run()
        _runner.wait()
        return _runner


__all__ = [
    "BuildConfig",
    "RuntimeConfig",
    "ContainerRunner",
    "ContainerImage",
    "AbcContainerTemplate",
]
