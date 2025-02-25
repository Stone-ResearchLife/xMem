import logging
import time
import tqdm
import docker
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, List, Tuple
from ures.docker import Container, Image, RuntimeConfig
from ures.string import unique_id


class Runtime(ABC):
    def __init__(self, containers: List[Container]):
        self._containers = containers

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def stop(self, *args, **kwargs):
        pass

    @abstractmethod
    def remove(self, *args, **kwargs):
        pass

    @abstractmethod
    def logs(self, *args, **kwargs):
        pass


class SequenceRuntime(Runtime):
    def run(self, *args, **kwargs):



class Containers:
    def __init__(self, image: Image, client: Optional[docker.DockerClient] = None):
        # Ensure that the image is an instance of the Image class
        assert isinstance(image, Image)
        # Ensure that the image exists locally
        assert image.exist is not False
        self._client = client or docker.from_env()
        self._image = image
        self._runtime_history: Dict[str, Dict[str, Union[Container, str]]] = {}

    @property
    def image(self) -> str:
        return self._image.get_fullname()
    
    @property
    def name(self):
        return f"{self._image.name}-instance-{unique_id()[:10]}"

    def get_container(self, new: bool = False) -> List[Tuple[str, Dict[str, Union[Container, str]]]]:
        if new:
            containers = list(filter(lambda x: x[1]["status"] == "Created", self._runtime_history.items()))
        else:
            containers = list(self._runtime_history.items())
        return containers

    def _default_config(self) -> RuntimeConfig:
        return RuntimeConfig(
            image_name=self.image,
            name=self.name,
            detach=True,
            remove=True
        )

    def _construct_config(self, **kwargs) -> RuntimeConfig:
        _conf = self._default_config()
        # key image_name must be removed from the kwargs
        # The only image can be used is the one that was
        # passed to the constructor
        kwargs.pop("image_name", None)
        # Initialize the runtime configuration
        _conf.model_copy(update=kwargs)
        return _conf

    def create(self, **kwargs) -> Container:
        _container = Container(image=self._image, client=self._client)
        _conf = self._construct_config(**kwargs)
        _container_name = _conf.name
        # create the container
        _container.create(config=_conf, tag=None)
        self._runtime_history[_container_name] = {
            "container": _container,
            "config": _conf,
            "status": _container.status,
            "exit_code": _container.exit_code
        }
        return _container


    def run(self, concurrent_run: bool = False) ->  List[Tuple[str, Dict[str, Union[Container, str]]]]:
        new_containers = self.get_container(new=True)
        for container_name, container in tqdm.tqdm(new_containers):
            _c: Container = container["container"]
            max_tries = 3
            for i in range(max_tries):
                _c.run()
                time.sleep(0.5)
                if _c.is_running:
                    break
                else:
                    logging.warning(f"Failed to run container {container_name} on try {i+1}/{max_tries}")

            if _c.is_running is False:
                logging.error(f"Failed to run container {container_name}")
            else:
                if concurrent_run is False:
                    _c.wait()

            container["status"] = _c.status
            container["exit_code"] = _c.exit_code
        return new_containers

