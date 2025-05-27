import fire
from paper_container.evaluations import Experiments
from paper_container.docker_cleanup import DockerCleanup


class APPs:
    """
    APPs class is responsible for initializing and managing various components of the application.

    Attributes:
        xmem: Instance of XMem for memory operations.
        xprofiler: Instance of XProfiler for performance profiling.
        tprofiler: Instance of TProfiler for thread profiling.

    """

    def __init__(self):
        self.experiments = Experiments()

    def cleanup(self):
        _cleanup = DockerCleanup()
        _cleanup.stopped_containers()
        _cleanup.dangling_images()


if __name__ == "__main__":
    fire.Fire(APPs)