import os.path
import uuid
import logging
from typing import Optional, Set, List, Generator, Union, Dict
from ..config import Config, default_setting
from .allocator import CachingAllocator, Device, Stream, Block


logger = logging.getLogger(__name__)


class SimulatedMemoryBlock:
    def __init__(self, block: Union["ActivityMemory", "MemoryBlock"]):
        self.block: Union["ActivityMemory", "MemoryBlock"] = block
        self.is_allocated = False
        self.allocate_block: Optional[Block] = None
        self.id = str(uuid.uuid4().hex)

    @property
    def size(self):
        return self.block.bytes

    @property
    def alloc_time(self):
        return self.block.alloc_time

    @property
    def free_time(self):
        return self.block.free_time

    def __repr__(self):
        return f"Sim Block: {self.size} - {self.alloc_time} - {self.free_time}"


class AllocatorSim:
    def __init__(
        self,
        max_allocated_memory_gb: Union[float, int] = None,
        config: Optional[Config] = None,
    ):
        if max_allocated_memory_gb is None:
            max_allocated_memory_gb = 8 * 1024**3
        else:
            max_allocated_memory_gb = max_allocated_memory_gb * 1024**3
        self._allocator = CachingAllocator(
            allowed_memory_maximum=max_allocated_memory_gb
        )
        self._config = config or default_setting

    def other_memory_sequence(self, data: List["MemoryBlock"]):
        _time_base_blocks = {}
        for mem_block in data:
            _sim_block = SimulatedMemoryBlock(mem_block)
            if _sim_block.alloc_time not in _time_base_blocks:
                _time_base_blocks[_sim_block.alloc_time] = []
            if (
                _sim_block.free_time is not None
                and _sim_block.free_time not in _time_base_blocks
            ):
                _time_base_blocks[_sim_block.free_time] = []
            _time_base_blocks[_sim_block.alloc_time].append(_sim_block)
            if _sim_block.free_time is not None:
                _time_base_blocks[_sim_block.free_time].append(_sim_block)
        _sorted_time_base_blocks = dict(
            sorted(_time_base_blocks.items(), key=lambda item: item[0])
        )
        return _sorted_time_base_blocks

    def simulate(
        self, data_analysis: Union[List["MemoryBlock"]], segment_plot: bool = False
    ) -> CachingAllocator:
        _log_dir = self._config.log_dir
        _snapshot_files = []
        _sim_blocks = self.other_memory_sequence(data_analysis)

        for _index, _blocks in enumerate(_sim_blocks.values()):
            if _index == 25:
                pass
            mlloc_first_blocks = sorted(_blocks, key=lambda x: x.is_allocated)
            try:
                for _sub_index, _block in enumerate(mlloc_first_blocks):
                    logger.info(f"Processing block {_index}/{_sub_index}")
                    _device = Device(index=0)
                    _stream = Stream(index=0)
                    if not _block.is_allocated and _block.allocate_block is None:
                        _alloc_block = self._allocator.malloc(
                            _device, _stream, _block.size
                        )
                        _block.allocate_block = _alloc_block
                        _block.is_allocated = True
                        _alloc_size = (
                            _alloc_block.size
                            if _alloc_block.requested_size is None
                            else _alloc_block.requested_size
                        )
                        if _block.size != _alloc_size:
                            print(
                                f"Block size is not equal to allocated size: {_block.size} != {_alloc_size}"
                            )
                            raise ValueError("Invalid block size")
                    elif _block.is_allocated and _block.allocate_block is not None:
                        self._allocator.free(_block.allocate_block)
                        _block.is_allocated = False
                        _block.allocate_block = None
                    else:
                        raise ValueError("Invalid block state")

                    if segment_plot:
                        _snapshot_dir = _log_dir.joinpath("segment_plots")
                        _snapshot_dir.mkdir(parents=True, exist_ok=True)
                        _snapshot_img = os.path.join(
                            _snapshot_dir, f"frame-{_index}-{_sub_index}.png"
                        )
                        _snapshot_files.append(_snapshot_img)
                        self._allocator.plot(_snapshot_img)
            except Exception as e:
                _snapshot_img = os.path.join(_log_dir, f"Last-frame.png")
                _snapshot_files.append(_snapshot_img)
                self._allocator.plot(_snapshot_img)
                print(
                    f"Memory allocation failed at block {_index}/{_sub_index}\nthe memory capture is saved at {_snapshot_img}"
                )
                _snapshot_img = os.path.join(_log_dir, f"Last-GPU-frame.png")
                self._allocator._gpu_device.plot(_snapshot_img)
                self._allocator.oom = True
                print(f"the memory capture is saved at {_snapshot_img}")
                break

        _result = self._allocator
        _result._last_sequence_history = _sim_blocks
        self._allocator = CachingAllocator(
            allowed_memory_maximum=_result.allowed_memory_maximum
        )
        return _result
