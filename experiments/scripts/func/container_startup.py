import argparse
import os
import pprint
from abc import ABC, abstractmethod
from pathlib import Path
from perf_estimator.utilis.utilis import fetch_temp_folder
from experiments.container.container import BuildConfig, RuntimeConfig, ContainerRunner


class InterfaceTemplate(ABC):
    @abstractmethod
    def build_config(self, args: argparse.Namespace) -> BuildConfig:
        pass

    @abstractmethod
    def runtime_config(self, args: argparse.Namespace) -> RuntimeConfig:
        pass


class AbcTemplate(InterfaceTemplate):
    @abstractmethod
    def build_config(self, args: argparse.Namespace) -> BuildConfig:
        pass

    def runtime_config(self, args: argparse.Namespace) -> RuntimeConfig:
        _params = {
            "image_name": args.image_name,
            "detach": args.detach,
            # "remove": args.remove,
        }
        if args.name is not None:
            _params["name"] = args.name
        if args.force_rebuild:
            _params["force_rebuild"] = args.force_rebuild
        if args.platform is not None:
            _params["platform"] = args.platform
        if args.cpus is not None:
            _params["cpus"] = args.cpus
        if args.gpus is not None:
            _params["gpus"] = args.gpus
        if args.memory is not None:
            _params["memory"] = args.memory
        if args.entrypoint is not None:
            _params["entrypoint"] = args.entrypoint
        if args.command is not None:
            _params["command"] = args.command
        if args.env is not None:
            _env = {}
            for env in args.env:
                key, value = env.split("=")
                _env[key] = value
            _params["env"] = _env
        if args.volumes is not None:
            _volumes = {}
            for volume in args.volumes:
                src, dest = volume.split(":")
                _volumes[src] = {
                    "bind": dest,
                    "mode": "rw"
                }
            _params["volumes"] = _volumes
        if args.subnet is not None and args.ipv4 is not None:
            _params["subnet"] = args.subnet
            _params["ipv4"] = args.ipv4
        return RuntimeConfig(**_params)


class PytorchModelTemplate(AbcTemplate):
    def __init__(self):
        # Build Config
        self._user = "pytorch"
        self._base_image = "pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime"
        self._python_dependencies = ["pydantic"]
        self._sys_dependencies = ["vim"]
        self._context_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._copies = [
            {
                "src": "perf_estimator",
                "dest": "perf_estimator",
            },
            {
                "src": "playground/trainer_test.py",
                "dest": "trainer.py",
            }
        ]
        self._entrypoint = ["python", "trainer.py"]
        # Runtime Config
        self._volumes = {
            str(Path().home().joinpath("pytorch_datasets")): {
                "bind": str(self._container_home_dir.joinpath("pytorch_datasets")),
                "mode": "rw"
            }
        }

    @property
    def _container_home_dir(self):
        return Path(f"/home/{self._user}")

    def build_config(self, args: argparse.Namespace) -> BuildConfig:
        _build_config = BuildConfig(
            base_image = self._base_image,
            python_dependencies = self._python_dependencies,
            sys_dependencies = self._sys_dependencies,
            labels = [("FRAMEWORK", "PyTorch")],
            uid=os.getuid(),
            user=self._user,
            copies = self._copies,
            context_dir = self._context_dir,
            entrypoint = self._entrypoint,
        )
        return _build_config
    
    def runtime_config(self, args: argparse.Namespace) -> RuntimeConfig:
        _params = {
            "image_name": args.image_name,
            "detach": args.detach,
            # "remove": args.remove,
            "volumes": self._volumes,
        }
        if args.name is not None:
            _params["name"] = args.name
        if args.force_rebuild:
            _params["force_rebuild"] = args.force_rebuild
        if args.platform is not None:
            _params["platform"] = args.platform
        if args.cpus is not None:
            _params["cpus"] = args.cpus
        if args.gpus is not None:
            _params["gpus"] = args.gpus
        if args.memory is not None:
            _params["memory"] = args.memory
        if args.entrypoint is not None:
            _params["entrypoint"] = args.entrypoint
        if args.command is not None:
            _params["command"] = args.command
        if args.env is not None:
            _env = {}
            for env in args.env:
                key, value = env.split("=")
                _env[key] = value
            _params["env"] = _env
        if args.volumes is not None:
            for volume in args.volumes:
                src, dest = volume.split(":")
                _params["volumes"][src] = {
                    "bind": dest,
                    "mode": "rw"
                }
        if args.subnet is not None and args.ipv4 is not None:
            _params["subnet"] = args.subnet
            _params["ipv4"] = args.ipv4
        return RuntimeConfig(**_params)


Templates = {
    "PytorchModel": PytorchModelTemplate,
}


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Runtime configuration for container execution")

    parser.add_argument("--image-name", type=str, default="model-runner",
                        help="Name of the image (default: model-runner)")
    parser.add_argument("--name", type=str, default=None,
                        help="Name of the container (default: None)")
    parser.add_argument("--force-rebuild", action="store_true", default=False,
                        help="Force rebuild the image (default: False)")
    parser.add_argument("--platform", type=str, default=None,
                        help="Platform for the base image (default: None)")
    parser.add_argument("--detach", action="store_true", default=False,
                        help="Run the container in detached mode (default: False)")
    parser.add_argument("--remove", action="store_true", default=False,
                        help="Remove the container after it is stopped (default: False)")
    parser.add_argument("--cpus", type=int, default=None,
                        help="Number of CPUs to allocate to the container (default: None)")
    parser.add_argument("--gpus", nargs='*', type=str, default=None,
                        help='List of GPUs to allocate to the container, for example: ["0", "1"] (default: None)')
    parser.add_argument("--memory", type=str, default=None,
                        help="Memory to allocate to the container, for example '2g' (default: None)")
    parser.add_argument("--entrypoint", nargs='*', type=str, default=None,
                        help="Entrypoint for the container (default: None)")
    parser.add_argument("--command", nargs='*', type=str, default=None,
                        help="Command for the container (default: None)")
    parser.add_argument("--env", nargs='*', type=str, default=None,
                        help="Environment variables for the container in key=value format (default: None)")
    parser.add_argument("--volumes", nargs='*', type=str, default=None,
                        help="Volumes to mount to the container in src:dest format (default: None)")
    parser.add_argument("--subnet", type=str, default=None,
                        help="Subnet for the container (default: None)")
    parser.add_argument("--ipv4", type=str, default=None,
                        help="IPv4 address for the container (default: None)")
    parser.add_argument("--temp-dir", type=Path, default=Path(fetch_temp_folder()),
                        help="Directory for the cache (default: fetched temp folder)")
    parser.add_argument("--template", type=str, default=list(Templates.keys())[0],
                        choices=Templates.keys(),
                        help="Template for the container execution (default: PytorchModelTemplate)")
    parser.add_argument("--build-only", action="store_true", default=False, help="Build the image only")

    return parser

