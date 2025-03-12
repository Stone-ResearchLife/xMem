import uuid
import logging
import sys
import copy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sortedcontainers import SortedSet
from typing import List, Optional, Dict, Union, Set
from pydantic import BaseModel, Field
from perf_estimator.data_structure import NonCircularDoublyLinkedNode as BiDirection
from perf_estimator.utilis.utilis import format_memory
from ._C2Python.llvm import *

logger = logging.getLogger(__name__)


class Device(BaseModel):
    index: int = Field(default=0, title="Index of the device")
    id: str = Field(default=uuid.uuid4().hex, title="ID of the device")


class Stream(BaseModel):
    index: int = Field(default=0, title="Index of the stream")
    id: str = Field(default=uuid.uuid4().hex, title="ID of the stream")


class Block(BiDirection):
    def __init__(
        self,
        device: Device,
        stream: Stream,
        size: int,
        address: Optional[int] = None,
        requested_size: int = 0,
        pool: Optional["BlockPool"] = None,
    ):
        """Create a block node.

        Args:
            device (Device): The device of the block.
            stream (Stream): The stream of the block.
            size (int): The size of the block.
            address (int): The address of the block, in hex
            requested_size (int): The required size of the block.
            pool (int): The pool of the block.
        """
        super().__init__(value=size)
        self.device = device
        self.stream = stream
        self.size = size
        self.requested_size = requested_size
        self.pool = pool
        self.address = address
        self.is_allocated = False
        self.seg_id = None
        logger.info(f"New block created: {self}")

    def __repr__(self):
        msg = f"Block(address={self.address}, size={self.size}, required_size={self.requested_size}) in S_{self.stream.index}/D_{self.device.index}"
        if self.seg_id is not None:
            msg = f"Segment {self.seg_id}|" + msg
        return msg

    @property
    def is_head(self) -> bool:
        return self.prev is None

    @property
    def is_split(self) -> bool:
        return self.prev is not None or self.next is not None

    def splice(self, memory_size: int) -> Optional["Block"]:
        """Split the block.

        Args:
            memory_size (int): The size of the memory.

        Returns:
            Block: The new block.
        """
        logger.info(f"A block({memory_size}) is split from {self}")
        if memory_size > self.size:
            raise ValueError(
                f"Cannot split a block of size {self.size} into a block of size {memory_size}"
            )
        if memory_size == self.size:
            return self
        new_block = Block(
            device=self.device,
            stream=self.stream,
            size=memory_size,
            address=self.address,
            pool=self.pool,
        )
        self.size -= memory_size
        if self.address is not None:
            self.address += memory_size
        self.insert_before(new_block)
        logger.info(f"New block created: {new_block}")
        logger.info(f"Remaining block: {self}")
        return new_block

    def coalesce(self) -> "Block":
        """Coalesce the block."""
        if self.next is not None and self.next.is_allocated is False:
            logger.info(
                f"Coalesce block {self.address} with next block {self.next.address}"
            )
            logger.info(f"Before coalescing-next node: {self.next}")
            logger.info(f"Before coalescing-current node: {self}")
            if self.pool is not None:
                logger.info(f"Remove block {self.next.address} from pool")
                self.pool.blocks.remove(self.next)
            _next = self.next
            self.next.remove()
            self.size += _next.size
            logger.info(f"After coalescing-current node: {self}")
        if self.prev is not None and self.prev.is_allocated is False:
            logger.info(
                f"Coalesce block {self.address} with previous block {self.prev.address}"
            )
            logger.info(f"Before coalescing-prev node: {self.prev}")
            logger.info(f"Before coalescing-current node: {self}")
            if self.pool is not None:
                logger.info(f"Remove block {self.prev.address} from pool")
                self.pool.blocks.remove(self.prev)
            _prev = self.prev
            self.prev.remove()
            self.size += _prev.size
            self.address = _prev.address
            logger.info(f"After coalescing-current node: {self}")
        self.is_allocated = False
        self.pool.insert_into_blocks(self)
        return self

    def __hash__(self):
        return hash((self.device.id, self.stream.id, self.size, self.address))

    def __eq__(self, other):
        return (
            self.device == other.device
            and self.stream == other.stream
            and self.address == other.address
            and self.size == other.size
        )

    def __lt__(self, other):
        if self.stream != other.stream:
            return id(self.stream) < id(other.stream)
        if self.size != other.size:
            return self.size < other.size
        if self.address is not None and other.address is not None:
            return self.address < other.address
        return False

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other


class BlockPool:
    def __init__(self, small: bool):
        self.is_small = small
        self.blocks: SortedSet[Block] = SortedSet(
            key=lambda block: (block.stream.index, block.size, block.address)
        )
        # self.unmapped = SortedSet(key=block_comparator_address)

    def _get_lower_bound_index(self, search_key: "Block") -> int:
        # Find the index of the first element that is >= search_key
        if search_key.address is None:
            idx = self.blocks.bisect_key_left(
                (search_key.stream.index, search_key.size)
            )
        else:
            idx = self.blocks.bisect_key_left(
                (search_key.stream.index, search_key.size, search_key.address)
            )
        return idx

    def lower_bound(self, search_key: "Block", releaseable=False) -> Optional["Block"]:
        # Find the index of the first element that is >= search_key
        idx = self._get_lower_bound_index(search_key)
        logger.debug(f"Lower bound index: {idx} by search key: {search_key}")

        if idx < len(self.blocks):
            if releaseable:
                _block: Block = self.blocks[idx]
                while not (_block.prev is None and _block.next is None):
                    idx += 1
                    if idx >= len(self.blocks):
                        return None
                    _block = self.blocks[idx]
            else:
                return self.blocks[idx]
        else:
            return None  # No element >= search_key

    def is_end_block(self, search_key: "Block") -> bool:
        if search_key is None:
            return True
        idx = self._get_lower_bound_index(search_key)
        return idx >= len(self.blocks)

    def is_begin_block(self, search_key: "Block") -> bool:
        idx = self._get_lower_bound_index(search_key)
        return idx == 0

    def insert_into_blocks(self, block: "Block") -> None:
        logger.debug(f"Insert block: {block} into blocks")
        self.blocks.add(block)


class AllocParams:
    def __init__(
        self,
        device: Device,
        stream: Stream,
        size: int,
        pool: BlockPool,
        alloc_size: int,
    ):
        self.search_key = Block(device=device, stream=stream, size=size)
        self.pool = pool
        self.alloc_size = alloc_size
        self.block = None
        self.err = None

    def device(self) -> Device:
        return self.search_key.device

    def stream(self) -> Stream:
        return self.search_key.stream

    def size(self) -> int:
        """The size is commonly a rounded-up value."""
        return self.search_key.size


class CUDAAllocatorConfig:
    kRoundUpPowerOfTwoIntervals = 16

    def __init__(self, conf: dict[str, any]):
        self._conf = conf
        self.max_split_size = sys.maxsize
        self.garbage_collection_threadhold = 0
        self.pinned_num_register_threads = 1
        self.expandable_segments = False
        self.pinned_use_cuda_host_register = False
        self._round_power2_divisions = [
            [2**i, 0] for i in range(0, self.kRoundUpPowerOfTwoIntervals)
        ]

    def roundup_power2_divisions(self, size: int) -> int:
        log_size = llvm_count_leading_zeros(size)

        # Intervals start at 1MB and end at 64GB
        interval_start = 63 - llvm_count_leading_zeros(1048576)  # 1MB = 2^20
        interval_end = 63 - llvm_count_leading_zeros(68719476736)  # 64GB = 2^36

        index = log_size - interval_start
        index = max(0, index)
        index = min(index, self.kRoundUpPowerOfTwoIntervals - 1)

        return self._round_power2_divisions[index][1]


class GPUDevice:
    def __init__(self, device: Device, max_memory: int):
        self.device = device
        self.stream = Stream(index=0)
        self.max_memory = max_memory
        self.pool = BlockPool(small=False)
        _block = Block(
            device=device,
            stream=self.stream,
            size=max_memory,
            pool=self.pool,
            address=0x0,
        )
        self.pool.insert_into_blocks(_block)
        self.active_blocks: Set[Block] = set()

    def malloc(self, size: int) -> Optional[Block]:
        search_key = Block(device=self.device, stream=self.stream, size=size)
        block = self.pool.lower_bound(search_key)
        if block is not None:
            self.pool.blocks.remove(block)
            if size < block.size:
                allocated_block = block.splice(size)
                self.pool.insert_into_blocks(block)
            else:
                allocated_block = block
            allocated_block.is_allocated = True
            self.active_blocks.add(allocated_block)
            return allocated_block
        logger.warning(f"OOM when allocating {size} bytes on device {self.device}")
        return None

    def free(self, block: Block) -> None:
        block.is_allocated = False
        self.active_blocks.remove(block)
        block.coalesce()
        self.pool.insert_into_blocks(block)

    def addr_free(self, address: int) -> None:
        for _block in self.active_blocks:
            if _block.address == address:
                self.free(_block)
                return

    def stats(self) -> dict:
        _stats = {
            "total": self.max_memory,
            "device": self.device.index,
            "stream": self.stream.index,
            "free": sum([block.size for block in self.pool.blocks]),
            "active": sum([block.size for block in self.active_blocks]),
            "blocks": list(self.active_blocks) + list(self.pool.blocks),
        }
        return _stats

    def plot(self, filename: Optional[str] = None):
        plot_memory_stats([self.stats()], filename=filename)


def plot_memory_stats(
    states: List[Dict[str, Union[int, str, Block, Device, Stream]]],
    filename: Optional[str] = None,
):
    """Plot memory states for all segments

    Examples:
        The input of states should be formed as below:
        states = [
            {
                "device": Device,
                "stream": Stream,
                "total: int,
                "free": int,
                "active": int,
                'blocks': [Block, Block, Block, ...]
            }
        ]

    Args:
        states (List[dict]): A list of memory states
        filename (str): The filename to save the plot

    Returns:
        None

    """
    num_states = len(states)

    # 创建子图，每个状态一个子图，垂直排列
    fig, axes = plt.subplots(num_states, 1, figsize=(10, 2 * num_states), sharex=False)

    if num_states == 1:
        axes = [axes]  # 确保 axes 是可迭代的

    for ax, state in zip(axes, states):
        total_size = state["total"]
        blocks: List[Block] = state["blocks"]
        blocks.sort(key=lambda x: x.address)

        begin_address = 0x0
        for index, block in enumerate(blocks):
            if index == 0:
                begin_address = block.address
            start = block.address - begin_address
            size = block.size
            is_allocated = block.is_allocated
            color = "red" if is_allocated else "green"

            ax.barh(0, size, left=start, height=0.5, color=color, edgecolor="black")

            # 标注内存块大小
            ax.text(
                start + size / 2,
                0,
                f"{format_memory(size)}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

        ax.set_xlim(0, total_size)
        ax.set_ylim(-0.5, 1)

        # Set x-axis label
        # ax.set_xlabel(f'Total Memory of Segment: ({format_memory(total_size)})')
        ax.get_xaxis().set_visible(False)

        # hidden the y-axis label
        ax.get_yaxis().set_visible(False)

        # 设置标题
        segment_id = blocks[0].seg_id or 0
        ax.set_title(
            state.get(
                "description",
                f"Segments({segment_id}) with Total Memory: ({format_memory(total_size)})",
            )
        )

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


class Stats(BaseModel):
    max_split_size: int = 0


class Trace:
    def __init__(self, segment_ref: Optional[Set["Block"]]):
        self._trace_history = []
        self._logger = logger
        self._segments = segment_ref
        self.max_segment_changes = [0]
        self.max_usage_changes = [0]

    def _record(self, block: "Block", action: str, use_size: bool = True):
        if isinstance(block, Block):
            _msg = self._gen_msg(block, use_size)
        else:
            _msg = "This is a invalid allocation"
        logger.info(f"[{action}] {_msg}")
        _record = {
            "action": action,
            "size": block.size,
            "requested_size": block.requested_size,
            "block": block,
            "msg": _msg,
            "segments": list(self._segments),
        }
        self._trace_history.append(_record)

    def _gen_msg(self, block: "Block", use_size: bool = True):
        if use_size:
            _size = getattr(block, "size")
        else:
            _size = getattr(block, "requested_size")

        return f"{None} Block from Segment {block.seg_id} for request {getattr(block, 'requested_size', None)}"

    def segment_alloc(self, block):
        self.max_segment_changes.append(self.max_segment_changes[-1] + block.size)
        self._record(block, "segment_alloc")

    def segment_free(self, block):
        self.max_segment_changes.append(self.max_segment_changes[-1] - block.size)
        self._record(block, "segment_free")

    def alloc(self, block):
        self.max_usage_changes.append(self.max_usage_changes[-1] + block.size)
        self._record(block, "alloc", False)

    def free_requested(self, block):
        self._record(block, "free_requested", False)

    def free_completed(self, block):
        self.max_usage_changes.append(self.max_usage_changes[-1] - block.size)
        self._record(block, "free_completed", True)


class CachingAllocator:
    kLargeBuffer = 20971520
    kMinBlockSize = 512
    kSmallSize = 1048576
    kSmallBuffer = 2097152
    kMinLargeAlloc = 10485760
    kRoundLarge = 2097152

    def __init__(self, allowed_memory_maximum: Optional[int] = None):
        self.small_pool: BlockPool = BlockPool(small=True)
        self.large_pool: BlockPool = BlockPool(small=False)
        self.active_blocks: Set[Block] = set()
        self.total_allocated_memory: int = 0
        self.allowed_memory_maximum: int = allowed_memory_maximum or 16 * 1024**3
        self.stats = Stats()
        self._cuda_config = CUDAAllocatorConfig(conf={})
        self.stats.max_split_size = self._cuda_config.max_split_size
        self._segments: Set[Block] = set()
        self._trace = Trace(self._segments)
        self._segment_count = 0
        self._gpu_device = GPUDevice(
            device=Device(index=0), max_memory=self.allowed_memory_maximum
        )
        self._oom = False

    @property
    def oom(self) -> bool:
        return self._oom

    @oom.setter
    def oom(self, value: bool):
        self._oom = value

    def roundup_power2_next_division(self, size: int, divisions: int) -> int:
        if divisions <= 1 or size <= 4:
            return size
        if llvm_is_power_of_2(size):
            return size
        power2_floor = llvm_power_of_2_floor(size)
        power2_division = power2_floor >> (63 - llvm_count_leading_zeros(divisions))
        if power2_division == 0:
            return power2_floor << 1
        round_size_floor = size & (~(power2_division - 1))
        return size if round_size_floor == size else round_size_floor + power2_division

    def round_up(self, size: int) -> int:
        if size <= self.kMinBlockSize:
            return self.kMinBlockSize
        else:
            divisions = self._cuda_config.roundup_power2_divisions(size)
            if divisions > 0 and size > (self.kMinBlockSize * divisions):
                # this part is implemented in roundup_power2_next_division method in PyTorch
                return self.roundup_power2_next_division(size, divisions)
            else:
                # Make sure the size is a multiple of kMinBlockSize
                return self.kMinBlockSize * (
                    (size + self.kMinBlockSize - 1) // self.kMinBlockSize
                )

    def get_pool(self, size: int) -> BlockPool:
        return self.small_pool if size <= self.kSmallSize else self.large_pool

    def get_allocation_size(self, size: int) -> int:
        if size <= self.kSmallSize:
            return self.kSmallBuffer
        elif size <= self.kMinLargeAlloc:
            return self.kLargeBuffer
        else:
            return self.kRoundLarge * (
                (size + self.kRoundLarge - 1) // self.kRoundLarge
            )

    def get_free_block(self, params: AllocParams) -> bool:
        _block = params.pool.lower_bound(params.search_key)
        if (
            (_block is None)
            or (params.size() < self.stats.max_split_size <= _block.size)
            or (
                params.size() >= self.stats.max_split_size
                and _block.size >= params.size() + self.kLargeBuffer
            )
        ):
            # this line is never be called when max_split_size used default value
            return False
        params.block = _block
        params.pool.blocks.remove(_block)
        return True

    def alloc_block(self, params: AllocParams) -> bool:
        _size = copy.deepcopy(params.alloc_size)
        if self.total_allocated_memory + _size > self.allowed_memory_maximum:
            return False
        gpu_block = self._gpu_device.malloc(_size)
        if gpu_block is None:
            return False
        self.total_allocated_memory += _size
        params.block = Block(
            device=params.device(),
            stream=params.stream(),
            size=_size,
            pool=params.pool,
            address=copy.deepcopy(gpu_block.address),
        )
        # add head of segment into segment_list
        params.block.seg_id = self._segment_count
        self._segment_count += 1
        self._segments.add(params.block)
        # todo trace here - segment_alloc
        self._trace.segment_alloc(params.block)
        return True

    def release_block(self, block: Block) -> None:
        self._gpu_device.addr_free(block.address)
        self._segments.remove(block)
        # todo trace here - segment_free
        self._trace.segment_free(block)
        self.total_allocated_memory -= block.size
        block.pool.blocks.remove(block)

    def release_available_cache_blocks(self, params: AllocParams):
        """Free one or more oversize blocks to the system allocator.  But only enough to satisfy the request."""
        if self.stats.max_split_size == sys.maxsize:
            # The release is always false when max_split_size is default value
            return False
        pool = params.pool
        key = Block(device=params.device(), stream=params.stream(), size=params.size())
        key.size = (
            self.stats.max_split_size
            if key.size < self.stats.max_split_size
            else key.size
        )
        # the reason why assign key.size to max_split_size is that the function only looks for oversize blocks
        # oversize means the block which the size is larger than max_split_size
        if len(pool.blocks) == 0:
            # this line replaces the origin code: if pool.is_begin_block(_block).
            # if _block is begin block and end block, simultaneously, the pool is empty.
            return False
        _block = pool.lower_bound(key, releaseable=True)
        if pool.is_end_block(_block):
            # There is no oversize block existing, so releasing multiple blocks, which size is less than the max_split_size
            # to satisfy the request.
            total_released = 0
            _block = pool.blocks[-1]
            while total_released < key.size and _block >= self.stats.max_split_size:
                total_released += _block.size
                if pool.is_begin_block(_block):
                    break
                elif _block.prev is not None or _block.next is not None:
                    _prev_index = pool.blocks.index(_block) - 1
                    _block = pool.blocks[_prev_index]
                    continue
                else:
                    _prev_index = pool.blocks.index(_block) - 1
                    _cur = _block
                    _block = pool.blocks[_prev_index]
                    self.release_block(_cur)
            if total_released < key.size:
                return False
        else:
            # Only release one oversize block to satisfy the request.
            # because key.size equals to max_split_size when it is less than max_split_size
            # and key.size is original size when it is larger than max_split_size
            # Thus, any block sought by lower_bound function means that this block can satisfy the request.
            self.release_block(_block)
        return True

    def release_blocks(self, pool: BlockPool):
        free_list = []
        for block in pool.blocks:
            if block.prev is None and block.next is None:
                free_list.append(block)
        for block in free_list:
            self.release_block(block)

    def release_cached_blocks(self):
        self.release_blocks(self.large_pool)
        self.release_blocks(self.small_pool)

    def should_split(self, block: Block, size: int) -> bool:
        remaining = block.size - size
        if block.pool.is_small:
            return remaining >= self.kMinBlockSize
        else:
            return (size < self.stats.max_split_size) and (remaining > self.kSmallSize)

    def alloc_found_block(
        self, params: AllocParams, origin_size: int, split_remainder: bool
    ) -> Optional[Block]:
        _size = copy.deepcopy(params.size())
        _device = params.device()
        _stream = params.stream()
        _pool = params.pool
        _block = params.block  # this block is fetched from pool. A result of search.
        remaining: Optional[Block] = None

        already_split = _block.is_split
        if split_remainder:
            remaining = _block
            segment_replace_need = False
            if remaining.is_head:
                self._segments.remove(remaining)
                segment_replace_need = True

            _block = remaining.splice(_size)
            _block.seg_id = remaining.seg_id

            remaining.is_allocated = False
            _pool.insert_into_blocks(remaining)
            # replace the head of segment by new block
            if segment_replace_need:
                self._segments.add(_block)

        _block.requested_size = origin_size
        _block.is_allocated = True
        self.active_blocks.add(_block)
        # todo trace here - block alloc
        self._trace.alloc(_block)

        return _block

    def malloc(
        self, device: Device, stream: Stream, origin_size: int
    ) -> Optional[Block]:
        _size = self.round_up(origin_size)
        _pool = self.get_pool(_size)
        _alloc_size = self.get_allocation_size(_size)
        _params = AllocParams(
            device=device, stream=stream, size=_size, pool=_pool, alloc_size=_alloc_size
        )
        _found = self.get_free_block(_params)
        if not _found:
            _found = self.alloc_block(_params)
            if not _found:
                self.release_available_cache_blocks(_params)
                _found = self.alloc_block(_params)
                if not _found:
                    self.release_cached_blocks()
                    _found = self.alloc_block(_params)

        if not _found:
            # start to handle OOm issue
            logger.warning(f"OOM when allocating {_size} bytes on device {device}")
            raise MemoryError(f"OOM when allocating {_size} bytes on device {device}")
        split_remainder: bool = self.should_split(_params.block, _params.size())
        return self.alloc_found_block(_params, origin_size, split_remainder)

    def free_block(self, block: Block):
        # todo trace here - free completed
        self._trace.free_completed(block)
        self.active_blocks.remove(block)
        pool = block.pool
        block = self.coalesce(block)
        pool.insert_into_blocks(block)
        return block

    def free(self, block: Block):
        # todo trace here - free requested
        self._trace.free_requested(block)
        block.is_allocated = False
        self.free_block(block)
        return block

    def coalesce(self, block: Block):
        p_block = block.prev
        segment_replace_need = False
        if (
            p_block is not None
            and p_block.is_allocated is False
            and p_block in list(self._segments)
        ):
            self._segments.remove(p_block)
            segment_replace_need = True
        if block.is_head:
            self._segments.remove(block)
            segment_replace_need = True

        block = block.coalesce()
        if segment_replace_need:
            self._segments.add(block)
        return block

    def get_segment_stats(self):
        seg_stats = []
        for seg in self._segments:
            if seg.prev is not None:
                print(f"Segment {seg.seg_id} is not head")
                continue
            else:
                total_blocks = []
                block = seg
                while block is not None:
                    total_blocks.append(block)
                    block = block.next

                _stats = {
                    "index": seg.seg_id,
                    "total": sum([block.size for block in total_blocks]),
                    "free": sum(
                        [
                            block.size
                            for block in total_blocks
                            if block.is_allocated is False
                        ]
                    ),
                    "active": sum(
                        [
                            block.size
                            for block in total_blocks
                            if block.is_allocated is True
                        ]
                    ),
                    "blocks": total_blocks,
                }
                seg_stats.append(_stats)
        return seg_stats

    def plot(self, file_name: Optional[str] = None):
        plot_memory_stats(self.get_segment_stats(), filename=file_name)

    def plot_memory_change(self):
        tensor_trace = self._trace.max_usage_changes
        segment_trace = self._trace.max_segment_changes

        # Create a new figure
        fig = go.Figure()

        # Add traces for tensor and segment memory usage
        fig.add_trace(
            go.Scatter(
                x=list(range(len(tensor_trace))),
                y=tensor_trace,
                mode="lines",
                name="Tensor Memory",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(segment_trace))),
                y=segment_trace,
                mode="lines",
                name="Segment Memory",
            )
        )

        # Update the layout with axis labels and title
        fig.update_layout(
            title="Memory Usage Over Time",
            xaxis_title="Time",
            yaxis_title="Memory Usage",
            width=1000,
            height=500,
        )

        # Display the figure
        fig.show()
