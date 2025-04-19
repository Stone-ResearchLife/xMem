import fire
from paper_container.apps import XMem, XProfiler
from paper_container.evaluations import Experiments


class APPs:
    """
    APPs class is responsible for initializing and managing various components of the application.

    Attributes:
        xmem: Instance of XMem for memory operations.
        xprofiler: Instance of XProfiler for performance profiling.
        tprofiler: Instance of TProfiler for thread profiling.

    """

    def __init__(self):
        self.xmem = XMem()
        self.xprofiler = XProfiler()
        self.experiments = Experiments()


if __name__ == "__main__":
    fire.Fire(APPs)
