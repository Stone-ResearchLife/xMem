import docker
import docker.errors
import docker.types
import docker.constants
import logging
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from pathlib import Path
from perf_estimator.utilis.utilis import fetch_temp_folder
from perf_estimator.utilis.network import verify_ip_in_subnet
from .image import ContainerImage, BuildConfig, ImageInfo


logger = logging.getLogger(__name__)


class RuntimeConfig(BaseModel):
    image_name: str = Field(default="model-runner", title="Image Name", description="Name of the image")
    name: Optional[str] = Field(default=None, title="Name", description="Name of the container")
    force_rebuild: bool = Field(default=False, title="Force Rebuild", description="Force rebuild the image")
    platform: Optional[str] = Field(default=None, title="Platform", description="Platform for the base image")
    detach: bool = Field(default=True, title="Detach", description="Run the container in detached mode")
    user: Optional[str] = Field(default=None, title="User", description="User for the container")
    remove: bool = Field(default=False, title="Remove", description="Remove the container after it is stopped")
    # for parameter cpus, the type is Optional[int] because the number of CPUs can be None
    cpus: Optional[int] = Field(default=None, title="CPUs", description="Number of CPUs to allocate to the container")
    # for parameter gpus, the type is List[str] because the GPU IDs are strings, for example ["0", "1"]
    gpus: Optional[List[str]] = Field(default=None, title="GPUs", description="List of GPUs to allocate to the container")
    # for parameter memory, the type is Optional[str] because the memory can be None, for example "2g"
    memory: Optional[str] = Field(default=None, title="Memory", description="Memory to allocate to the container")
    entrypoint: Optional[List[Union[str, float, int, Path]]] = Field(default=None, title="Entrypoint", description="Entrypoint for the container")
    command: Optional[List[Union[int, float, str, Path]]] = Field(default=None, title="Command", description="Command for the container")
    env: Optional[Dict[str, str]] = Field(default=None, title="Environment", description="Environment variables for the container")
    volumes: Optional[Dict[str, Dict[str, str]]] = Field(default=None, title="Volumes", description="Volumes to mount to the container")
    subnet: Optional[str] = Field(default=None, title="Subnet", description="Subnet for the container")
    ipv4: Optional[str] = Field(default=None, title="IPv4", description="IPv4 address for the container")
    temp_dir: Path = Field(default=Path(fetch_temp_folder()), title="Cache Directory", description="Directory for the cache")
    build_config: Optional[BuildConfig] = Field(default=None, title="Build Config", description="Build configuration for the image")

    def output_dir(self) -> Path:
        _cache_dir = self.temp_dir.joinpath("output")
        _cache_dir.mkdir(parents=True, exist_ok=True)
        return _cache_dir
    
    def add_volume(self, host_path: Union[str, Path], container_path: Union[str, Path], mode: str = "rw"):
        logger.info(f"Adding volume {host_path} to {container_path} with mode {mode}")
        if self.volumes is None:
            self.volumes = {}
        self.volumes[str(host_path)] = {
            "bind": str(container_path),
            "mode": mode
        }
    
    def add_env(self, key: str, value: str):
        logger.info(f"Adding environment variable {key} with value {value}")
        if self.env is None:
            self.env = {}
        self.env[key] = value
    
    def set_subnet(self, subnet: str, ipv4: str):
        logger.info(f"Setting subnet {subnet} with ipv4 {ipv4}")
        try:
            docker_network = docker.from_env().networks.get(subnet)
        except docker.errors.NotFound:
            raise ValueError(f"Docker subnet {subnet} not found")
        else:
            if not verify_ip_in_subnet(ipv4, docker_network.attrs["IPAM"]["Config"][0]["Subnet"]):
                raise ValueError(f"Invalid IP address {ipv4} for network {subnet}")
            else:
                self.subnet = subnet
                self.ipv4 = ipv4

    def to_json(self):
        _params = {
            "image": self.image_name,
            "auto_remove": self.remove,
            "detach": self.detach,
            # "user": f"{os.getuid()}:{os.getgid()}",
        }
        if self.cpus is not None:
            _params["cpuset_cpus"] = self.cpus
        if self.gpus is not None:
            _params["device_requests"] = [
                docker.types.DeviceRequest(
                    driver="nvidia",
                    device_ids=self.gpus,
                    capabilities=[["gpu"]]
                )
            ]
        if self.memory is not None:
            _params["mem_limit"] = self.memory
        if self.entrypoint is not None:
            _params["entrypoint"] = [str(_entrypoint) for _entrypoint in self.entrypoint]
        if self.command is not None:
            _params["command"] = [str(_command) for _command in self.command]
        if self.volumes is not None:
            _params["volumes"] = self.volumes
        if self.env is not None:
            _params["environment"] = self.env
        if self.name is not None:
            _params["name"] = self.name
        if self.user is not None:
            _params["user"] = self.user
        return _params


class ContainerRunner:
    def __init__(self, config: Optional[RuntimeConfig] = None):
        self._client = docker.from_env()
        self._config = config or RuntimeConfig()
        self._container = None

    @property
    def is_running(self):
        return self.status == "running"
        
    @property
    def status(self):
        try:
            status = self._client.containers.get(self._container.id).status
        except docker.errors.NotFound:
            status = "removed"
        return status

    @property
    def exit_code(self):
        try:
            exit_code = self._client.containers.get(self._container.id).attrs["State"]["ExitCode"]
        except docker.errors.NotFound:
            exit_code = None
        return exit_code

    def _get_image(self) -> ImageInfo:
        _build_config = self._config.build_config or BuildConfig()
        _build_data_dir = self._config.temp_dir.joinpath("build")
        _build_data_dir.mkdir(parents=True, exist_ok=True)
        return self.build_image(
            name=self._config.image_name,
            config=_build_config,
            force=self._config.force_rebuild,
            temp_dir=_build_data_dir
        )

    @staticmethod
    def build_image(
            name: str,
            config: Optional[BuildConfig] = None,
            force: bool = False,
            temp_dir: Optional[Union[Path, str]] = None
    ) -> ImageInfo:
        _config = config or BuildConfig()
        _temp_dir = Path(temp_dir) or Path(fetch_temp_folder())
        _temp_dir.mkdir(parents=True, exist_ok=True)
        _image = ContainerImage(
            image_name=name,
            config=_config,
            temp_dir=_temp_dir,
            rebuild=force,
        )
        return _image.get_image()

    
    def create(self):
        _image = self._get_image()
        if _image.name != self._config.image_name:
            logger.info(f"Image name is different from the config: {_image.name} != {self._config.image_name}")
            logger.info(f"Updating image name to {_image.name}")
            self._config.image_name = _image.name
        params = self._config.to_json()
        logger.info(f"Creating container with params {params}")
        _container = self._client.containers.create(**params)
        if self._config.subnet is not None:
            net = self._client.networks.get(self._config.subnet)
            net.connect(_container, ipv4_address=self._config.ipv4)
        return _container
    
    def stop(self):
        self._container.stop()

    def remove(self):
        self._container.remove()

    def logs(self):
        return self._container.logs()

    def wait(self):
        self._container.wait()
    
    def run(self):
        if self._container is not None:
            self.stop()
            self.remove()
            if self.status == "removed":
                self._container = None
        _container = self.create()
        _container.start()
        logger.info(f"Container {self._config.name} is running")
        self._container = _container

