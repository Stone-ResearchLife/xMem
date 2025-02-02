import json
import os
import docker
import docker.models.containers
import docker.errors
import docker.types
import logging
import copy
from pathlib import Path
from docker.models.images import Image
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple, Union
from perf_estimator.utilis.utilis import fetch_temp_folder


logger = logging.getLogger(__name__)


class BuildConfig(BaseModel):
    base_image: str = Field(
        default="python:3.10-slim",
        title="Base Image",
        description="Base image for the container",
    )
    platform: Optional[str] = Field(
        default=None, title="Platform", description="Platform for the base image"
    )
    python_dependencies: Optional[List[str]] = Field(
        default=None,
        title="Python Dependencies",
        description="Python dependencies to be installed",
    )
    sys_dependencies: Optional[List[str]] = Field(
        default=None,
        title="System Dependencies",
        description="System dependencies to be installed",
    )
    labels: Optional[List[Tuple[str, str]]] = Field(
        default=None, title="Labels", description="Labels for the image"
    )
    uid: Optional[int] = Field(
        default=None, title="User ID", description="User ID for the container"
    )
    user: Optional[str] = Field(
        default=None, title="User", description="User for the container"
    )
    entrypoint: Optional[List[str]] = Field(
        default=None, title="Entrypoint", description="Entrypoint for the container"
    )
    cmd: Optional[List[str]] = Field(
        default=None, title="Command", description="Command for the container"
    )
    # The data structure for environment is a dictionary with the keys being the environment variable names
    environment: Optional[Dict[str, str]] = Field(
        default=None,
        title="Environment",
        description="Environment variables for the container",
    )
    # The data structure for copies is a list of dictionaries with the keys "src", "dest", and "mode"
    copies: Optional[List[Dict[str, str]]] = Field(
        default=None, title="Copies", description="Files to be copied to the container"
    )
    context_dir: Union[str, Path] = Field(
        default=Path().cwd(),
        title="Context Directory",
        description="Directory for the build context",
    )

    def add_label(self, key: str, value: str):
        logger.info(f"Adding label {key} with value {value}")
        if self.labels is None:
            self.labels = []
        self.labels.append((key, value))

    def add_copy(self, src: str, dest: str):
        logger.info(f"Adding copy {src} to {dest}")
        if self.copies is None:
            self.copies = []
        self.copies.append({"src": src, "dest": dest})

    def add_environment(self, key: str, value: str):
        logger.info(f"Adding environment variable {key} with value {value}")
        if self.environment is None:
            self.environment = {}
        self.environment[key] = value

    def set_context_dir(self, context_dir: Union[str, Path]):
        logger.info(f"Setting context directory to {context_dir}")
        if isinstance(context_dir, str):
            context_dir = Path(context_dir)
        if not context_dir.is_dir():
            raise ValueError(f"Context directory {context_dir} is not a directory")
        self.context_dir = context_dir


class DockerfileConstructor:
    def __init__(self, config: Optional[BuildConfig] = None):
        self._config = config or BuildConfig()
        self._dockerfile_content = []
        self.build()

    @property
    def home_dir(self):
        return f"/home/{self._config.user}"

    def _insert_command_line(self, command: str):
        logger.debug(f"Inserting command line: {command}")
        self._dockerfile_content.append(command)

    def build(self):
        self._dockerfile_content = []
        self._set_based_image()
        self._set_labels()
        self._set_environment()
        self._set_system_dependencies()
        self._set_python_dependencies()
        self._set_user()
        self._set_copy()
        self._set_entrypoint()
        self._set_cmd()

    def save(self, dest: Union[str, Path]):
        _dest = Path(dest) if isinstance(dest, str) else dest
        if _dest.is_dir():
            _dest = _dest / "Dockerfile"
        logger.info(f"Saving Dockerfile to {_dest}")
        _dest.parent.mkdir(parents=True, exist_ok=True)
        with open(_dest, "w") as f:
            f.write("\n".join(self._dockerfile_content))

    def _set_based_image(self):
        _config = self._config
        base_image = (
            _config.base_image
            if _config.platform is None
            else f"--platform={_config.platform} {_config.base_image}"
        )
        self._insert_command_line(f"FROM {base_image}")

    def _set_labels(self):
        _labels = []
        if self._config.labels is not None:
            _labels = _labels + self._config.labels
        for _label in _labels:
            self._insert_command_line(f'LABEL "{_label[0]}"="{_label[1]}"')

    def _set_homedir(self):
        self._insert_command_line(f"ARG HOME_DIR={self.home_dir}")

    def _set_user(self):
        if self._config.user is not None:
            self._set_homedir()
            self._insert_command_line(f"ARG USER_NAME={self._config.user}")
            if self._config.uid is not None:
                self._insert_command_line(f"ARG UID={self._config.uid}")
                self._insert_command_line(
                    f"RUN useradd -m -u $UID -s /bin/bash -d $HOME_DIR $USER_NAME"
                )
                self._insert_command_line(
                    f"RUN chown -R $USER_NAME:$USER_NAME $HOME_DIR"
                )
            self._insert_command_line(f"USER $USER_NAME")
            self._insert_command_line(f"WORKDIR $HOME_DIR")

    def _set_system_dependencies(self):
        _system_dependencies = []
        package_manager = "apt"
        if self._config.sys_dependencies is not None:
            _system_dependencies = _system_dependencies + self._config.sys_dependencies
        if len(_system_dependencies) > 0:
            self._insert_command_line(
                f"RUN {package_manager} update && {package_manager} install -y {' '.join(_system_dependencies)} && {package_manager} clean && rm -rf /var/lib/apt/lists/*"
            )

    def _set_python_dependencies(self):
        _python_dependencies = []
        if self._config.python_dependencies is not None:
            _python_dependencies = (
                _python_dependencies + self._config.python_dependencies
            )
        if len(_python_dependencies) > 0:
            self._insert_command_line(
                f"RUN pip install --upgrade pip && pip install --no-cache-dir {' '.join(_python_dependencies)} && rm -rf /tmp/* /var/tmp/*"
            )

    def _set_entrypoint(self):
        _entrypoint = self._config.entrypoint
        if _entrypoint is not None:
            self._insert_command_line(f"ENTRYPOINT {json.dumps(_entrypoint)}")

    def _set_cmd(self):
        _cmd = self._config.cmd
        if _cmd is not None and self._config.entrypoint is None:
            self._insert_command_line(f"CMD {json.dumps(_cmd)}")

    def _set_copy(self):
        _copies = self._config.copies
        if _copies is not None:
            for _copy in _copies:
                src = _copy["src"]
                dest = _copy["dest"]
                if dest[0] != "/":
                    if dest[0] == "." and dest[1] == "/":
                        dest = f"$HOME_DIR/{dest[2:]}"
                    else:
                        dest = f"$HOME_DIR/{dest}"
                self._insert_command_line(f"COPY --chown=$USER_NAME {src} {dest}")

    def _set_environment(self):
        _environment = self._config.environment
        if _environment is not None:
            for key, value in _environment.items():
                self._insert_command_line(f"ENV {key}={value}")


class ImageInfo:
    def __init__(self, image: Image):
        self._image = image

    @property
    def image(self) -> Image:
        return self._image

    @property
    def id(self) -> str:
        return self._image.id

    @property
    def architecture(self) -> str:
        return self._image.attrs["Architecture"]

    @property
    def image_size(self) -> int:
        """
        get the size of the image in MB
        :return: int
        """
        return int(self._image.attrs["Size"])

    @property
    def labels(self) -> dict:
        return self._image.labels

    @property
    def name(self) -> str:
        """
        Returns:
            str: The name of the image, e.g., "model-runner:latest"
        """
        return self._image.tags[0]

    def info(self):
        print(
            f"\033[1;33m====================================== Image Info ===============================================\033[0m"
        )
        print(f"Name: {self.name}")
        print(f"Image ID: {self.id}")
        print(f"Architecture: {self.architecture}")
        print(f"Image Size: {self.image_size} MB")
        print(f"Labels: {self.labels}")


class ContainerImage:
    def __init__(
        self,
        image_name: str,
        config: Optional[BuildConfig] = None,
        rebuild: bool = False,
        temp_dir: Optional[Path] = None,
    ):
        self._image_name = image_name
        self._temp_dir = temp_dir or fetch_temp_folder()
        self._rebuild = rebuild
        self._client = docker.from_env()
        self._build_config = config or BuildConfig()

    def cache_dir(self) -> Path:
        _cache_dir = self._temp_dir
        _cache_dir.mkdir(parents=True, exist_ok=True)
        return _cache_dir

    def get_image(self) -> ImageInfo:
        _is_build_needed = self.is_build_needed()
        if _is_build_needed:
            logger.warning(
                f"\033[1;33m====================================== Image Not Found, Start Building ===============================================\033[0m"
            )
            _base_image = self._create_base_image()
            _runtime_image = self._create_runtime_image(_base_image.name)
        else:
            logger.info(
                f"\033[1;33m====================================== Image Found, Skip Building ===============================================\033[0m"
            )
            _runtime_image = self._get_image()
        return _runtime_image

    def get_image_name(self, name: Optional[str] = None) -> tuple[str, str]:
        """Get the image name and tag

        Returns:
            tuple[str, str]: The image name and tag

        """
        _image_name = name or self._image_name
        logger.debug(
            f"Getting image name and tag for {_image_name}, input name: {name}"
        )
        _split_image_name = str(_image_name).split(":")
        if len(_split_image_name) == 1:
            return _split_image_name[0], "latest"
        return _split_image_name[0], _split_image_name[1]

    def _get_image(self, image_name: Optional[str] = None) -> Optional[ImageInfo]:
        _image_name = image_name or self._image_name
        _name, _tag = self.get_image_name(_image_name)
        _research = self._client.images.list(name=f"{_name}:{_tag}")
        if len(_research) > 0:
            logger.info(f"{len(_research)} images found for {_name}:{_tag}")
            logger.info(f"The 1st image is {_research[0]} picked")
            return ImageInfo(_research[0])
        logger.info(f"Image {_image_name} is not found")
        return None

    def is_build_needed(self) -> bool:
        if self._rebuild:
            return True
        logger.debug("Checking if build is needed")
        _image = self._get_image()
        if _image is None:
            return True
        return False

    def pull_image(self, image_name: Optional[str] = None) -> ImageInfo:
        _base_image = image_name or self._build_config.base_image
        _name, _tag = self.get_image_name(_base_image)
        logger.debug(f"Pulling base image {_name}:{_tag} from Docker Hub")
        _image = self._client.images.pull(_name, tag=_tag)
        logger.info(f"Base image {_name}:{_tag} is pulled successfully!")
        return ImageInfo(_image)

    def _create_base_image(self) -> ImageInfo:
        # Start build a base image
        _build_config = BuildConfig(
            base_image=self._build_config.base_image,
            python_dependencies=self._build_config.python_dependencies,
            sys_dependencies=self._build_config.sys_dependencies,
        )

        _dockerfile_constructor = DockerfileConstructor(_build_config)
        _dockerfile_path = self.cache_dir() / "Dockerfile-base"
        _dockerfile_constructor.save(_dockerfile_path)
        print(
            f"\033[1;33m====================================== Start Building Base Image ===============================================\033[0m"
        )
        print(f"Dockerfile Path: {_dockerfile_path}")
        base_image_name = f"base-{self._image_name}"
        _name, _tag = self.get_image_name(base_image_name)
        base_image_name = f"{_name}:{_tag}"
        return self._build_image(
            _dockerfile_path, base_image_name, _build_config.context_dir
        )

    def _create_runtime_image(self, base_image: Optional[str] = None) -> ImageInfo:
        # Start build a runtime image
        _build_config = copy.deepcopy(self._build_config)
        _build_config.base_image = base_image or self._build_config.base_image
        _build_config.sys_dependencies = None
        _build_config.python_dependencies = None
        if _build_config.user is None:
            _build_config.user = "runner"
        if _build_config.uid is None:
            _build_config.uid = os.getuid()

        _dockerfile_constructor = DockerfileConstructor(_build_config)
        _dockerfile_path = self.cache_dir() / "Dockerfile-runtime"
        _dockerfile_constructor.save(_dockerfile_path)
        print(
            f"\033[1;33m====================================== Start Building Runtime Image ===============================================\033[0m"
        )
        print(f"Dockerfile Path: {_dockerfile_path}")
        return self._build_image(
            _dockerfile_path, self._image_name, _build_config.context_dir
        )

    def _build_image(self, dockerfile: Path, tag: str, context: Path) -> ImageInfo:
        args = {
            "path": str(context),
            "tag": tag,
            "dockerfile": str(dockerfile),
        }
        print(f"Build Image Args: {args}")
        if self._rebuild:
            args["nocache"] = True
        logger.info(f"Building an image {tag} with args: {args}")
        try:
            image, build_log = self._client.images.build(**args)
        except docker.errors.BuildError as e:
            logger.error(f"Failed to build image {tag}")
            for log in e.build_log:
                logger.error(log)
            raise e
        for line in build_log:
            logger.debug(line)
        logger.info(f"Image {tag} is built successfully!")
        image_info = ImageInfo(image)
        print(
            f"\033[1;33m====================================== {image_info.name} is built successfully! ===============================================\033[0m"
        )
        image_info.info()
        return image_info
