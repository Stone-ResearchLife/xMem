import re
from typing import List, Dict, Optional
from perf_estimator.data_structure.memory import MemoryBlock as AbstractMemoryBlock


class SnapshotTraceBlock:
    def __init__(self, data: dict):
        self._data = data

    @property
    def action(self) -> str:
        """

        Examples:
            'alloc'  # memory allocated
            'free_requested', # the allocated received a call to free memory
            'free_completed', # the memory that was requested to be freed is now
                            # able to be used in future allocation calls
            'segment_alloc', # the caching allocator ask cudaMalloc for more memory
                            # and added it as a segment in its cache
            'segment_free',  # the caching allocator called cudaFree to return memory
                            # to cuda possibly trying free up memory to
                            # allocate more segments or because empty_caches was called
            'oom',          # the allocator threw an OOM exception. 'size' is
                            # the requested number of bytes that did not succeed
            'snapshot'      # the allocator generated a memory snapshot
                            # useful to coorelate a previously taken
                            # snapshot with this trace

        Returns:
            str: Action type of the block

        """
        return self._data["action"]

    @property
    def address(self) -> str:
        return str(self._data["addr"])

    @property
    def bytes(self) -> int:
        return self._data["size"]

    @property
    def time(self) -> int:
        return self._data["time_us"]

    @property
    def frames(self) -> List[Dict]:
        return self._data["frames"]


class MemoryBlock(AbstractMemoryBlock):
    def __init__(self, block: SnapshotTraceBlock):
        self._start: SnapshotTraceBlock = block
        self._end: Optional[SnapshotTraceBlock] = None

    @property
    def address(self) -> str:
        return str(self._start.address)

    @property
    def bytes(self) -> int:
        return self._start.bytes

    @property
    def alloc_time(self) -> int:
        return self._start.time

    @property
    def free_time(self) -> Optional[int]:
        return self._end.time if self.is_freed else None

    @property
    def duration(self) -> Optional[int]:
        return self.free_time - self.alloc_time if self.is_freed else None

    @property
    def is_freed(self) -> bool:
        return self._end is not None

    def set_end_block(self, value):
        assert value.address == self.address
        self._end = value


class ActivityMemory(MemoryBlock):
    def __init__(self, block: SnapshotTraceBlock):
        super().__init__(block)
        self._ops: Optional[List[str]] = None
        self._cublas_used = False

    @property
    def ops_name(self) -> str:
        return self.get_ops_stack()[0]

    def __repr__(self):
        return f"{self.ops_name}({self._cublas_used}|{self.is_freed}|{self.address}): bytes: {self.bytes}, start: {self.alloc_time}, stop: {self.free_time}"

    def get_ops_stack(self):
        if self._ops is None:
            ops = []
            for frame in reversed(self._start.frames):
                patterns = [
                    "at::_ops::[a-zA-Z0-9_]*::call",
                    "torch::autograd::generated::[a-zA-Z0-9_]*[0-9]{1,2}::apply",
                ]
                for pattern in patterns:
                    if re.match(pattern, frame["name"]):
                        _search = re.search(pattern, frame["name"])
                        layer_name = str(_search.group()).split("::")[-2]
                        ops.append(layer_name)
                if "Blas.cpp" == frame["filename"]:
                    self._cublas_used = True
            self._ops = ops
        return self._ops


class SegmentMemory(MemoryBlock):
    def __init__(self, block: SnapshotTraceBlock):
        super().__init__(block)

    def __repr__(self):
        return f"Segment Block({self.is_freed}|{self.address}): {self.bytes}, start: {self.alloc_time}, stop: {self.free_time}"
