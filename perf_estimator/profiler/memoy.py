from typing import Any, Dict, List, Optional, Union
from bisect import bisect_left, bisect_right
from perf_estimator.data_structure.memory import MemoryBlock as AbstractMemoryBlock
from .node import CpuInstantNode


class MemoryBlock(AbstractMemoryBlock):
    def __init__(self, node: CpuInstantNode):
        # debug
        if int(node.bytes) < 0:
            print(CpuInstantNode)
            raise RuntimeError(f"Memory block {node} is a negative value")
        self._start: CpuInstantNode = node
        self._end: Optional[CpuInstantNode] = None
        self._forward = False
        self._backward = False
        self._comments = []

    @property
    def address(self) -> str:
        return str(self._start.address)

    @property
    def bytes(self) -> int:
        return self._start.bytes

    @property
    def alloc_time(self) -> int:
        return self._start.start_time

    @property
    def free_time(self) -> Optional[int]:
        return self._end.end_time if self._end is not None else None

    @property
    def is_freed(self) -> bool:
        return self._end is not None

    @property
    def is_forward(self) -> bool:
        return self._forward

    @property
    def is_backward(self) -> bool:
        return self._backward

    @is_backward.setter
    def is_backward(self, value):
        pass

    @is_forward.setter
    def is_forward(self, value):
        pass

    @property
    def comments(self) -> List[str]:
        return list(set(self._comments))

    @property
    def duration(self) -> Optional[int]:
        return (
            (self.free_time - self.alloc_time) if self.free_time is not None else None
        )

    def __repr__(self):
        if self.is_forward:
            _mem_type = "forward"
        elif self.is_backward:
            _mem_type = "backward"
        else:
            _mem_type = "other"
        return f"Memory({_mem_type}|{self.address}|{'free' if self.is_freed else 'persis'}): Bytes:{self.bytes}, Start:{self.alloc_time}, End:{self.free_time}, Comments: {' '.join(self.comments)}"

    def set_forward(self):
        self._forward = True
        self._backward = False

    def set_backward(self):
        self._forward = False
        self._backward = True

    def set_free_node(self, node: CpuInstantNode):
        if str(node.address) != self.address:
            raise ValueError(
                "The address of the free node is not the same as the current block."
                f"Original address: {self.address} != free address: {node.address}"
            )
        if self.bytes != (node.bytes * -1):
            raise ValueError(
                f"Time {self.alloc_time}, the amount of memory freed is not the same as the current block."
                f"Original Size: {self.bytes} != free Size: {node.bytes}"
            )
        self._end = node


class MemoryActivity:
    def __init__(self, data: List[Dict[str, Any]]):
        self._data: Dict[int, List[MemoryBlock]] = self._build_up(data)

    @property
    def activities(self) -> Dict[int, List[MemoryBlock]]:
        return self._data

    def _build_up(self, data: List[Dict[str, Any]]) -> Dict[int, List[MemoryBlock]]:
        _address: Dict[str, MemoryBlock] = {}
        _activities: Dict[int, List[MemoryBlock]] = {}
        for trace in data:
            # To filter out the non-memory related events
            if trace.get("cat", None) not in ["cpu_instant_event"]:
                continue
            # store memory activities in the form of MemoryBlock based on the address
            # the block will be stored into activities when it is freed
            # all retaining memory will be added to activities after the end of the trace
            _node = CpuInstantNode(trace)
            _node_addr = str(_node.address)
            if _node_addr not in _address.keys():
                _mem_block = MemoryBlock(_node)
                _address[_node_addr] = _mem_block
            else:
                _address[_node_addr].set_free_node(_node)
                _mem_block = _address.pop(_node_addr)
                if _mem_block.alloc_time not in _activities.keys():
                    _activities[_mem_block.alloc_time] = []
                _activities[_mem_block.alloc_time].append(_mem_block)
        for value in _address.values():
            if value.alloc_time not in _activities.keys():
                _activities[value.alloc_time] = []
            _activities[value.alloc_time].append(value)

        return _activities

    def search_activities_in_time_range(
        self, start: Union[int, float], end: Union[int, float]
    ) -> List[MemoryBlock]:
        """The function is built upon the binary search algorithm to find all the operator nodes within a specified time range.

        Args:
            start (int): the start time of the time range.
            end (int): the end time of the time range.

        Returns:
            List[OperatorNode]: a list of operator nodes within the specified time range.
        """
        sorted_timestamps = sorted(self.activities.keys())

        # To find the start and end positions of the time using binary search
        start_idx = bisect_left(sorted_timestamps, start)
        end_idx = bisect_right(sorted_timestamps, end)

        # To collect all values within a specified time range
        result = []
        for i in range(start_idx, end_idx):
            for item in self._data[sorted_timestamps[i]]:
                result.append(item)

        result.sort(key=lambda x: x.alloc_time)
        return result
