import fire
from xmem_container.apps import XMem, XProfiler, TProfiler
from xmem_container.evaluations import HFTComparsion


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
        self.tprofiler = TProfiler()
        self.hftcomp = HFTComparsion()


if __name__ == "__main__":
    fire.Fire(APPs)
