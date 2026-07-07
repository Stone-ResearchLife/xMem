import json
import multiprocessing
import re
import time
from types import SimpleNamespace
import copy
import logging
from uuid import uuid4
import plotly.graph_objects as go
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Tuple
from bisect import bisect_left, bisect_right
from tqdm import tqdm
from perf_estimator.utilis.utilis import format_memory
from . import StackNode, OperatorNode, CpuInstantNode, MemoryBlock, ProfilerNode


logger = logging.getLogger(__name__)

try:
    import msgspec

    class TraceEvent(msgspec.Struct, gc=False):
        """Schema-decoded trace event. `args` stays as raw JSON bytes and is
        only decoded for the small subset of events the analysis touches."""

        name: str = ""
        cat: str = ""
        ph: str = ""
        ts: Union[int, float] = 0
        dur: Union[int, float] = 0
        pid: Union[int, str] = 0
        tid: Union[int, str] = 0
        args: msgspec.Raw = msgspec.Raw(b"{}")

    class _TraceFile(msgspec.Struct, gc=False):
        traceEvents: List[TraceEvent] = []

    class _SeqArgs(msgspec.Struct, gc=False):
        seq_number: Optional[Union[int, float]] = msgspec.field(
            name="Sequence number", default=None
        )

    _TRACE_DECODER = msgspec.json.Decoder(_TraceFile)
    _SEQ_DECODER = msgspec.json.Decoder(_SeqArgs)
    _ARGS_DECODER = msgspec.json.Decoder()
except ImportError:  # fall back to stdlib json parsing
    msgspec = None


def event_to_dict(event) -> dict:
    """Convert a decoded trace event to the plain dict the node classes use."""
    args = event.args
    if not isinstance(args, dict):  # msgspec.Raw bytes
        args = _ARGS_DECODER.decode(args)
    return {
        "name": event.name,
        "cat": event.cat,
        "ph": event.ph,
        "ts": event.ts,
        "dur": event.dur,
        "pid": event.pid,
        "tid": event.tid,
        "args": args,
    }


def event_seq_number(event) -> Optional[Union[int, float]]:
    """Read args["Sequence number"] without decoding the full args payload."""
    args = event.args
    if isinstance(args, dict):
        return args.get("Sequence number", None)
    return _SEQ_DECODER.decode(args).seq_number


class Layer:
    def __init__(self, node: StackNode):
        assert isinstance(node, StackNode)
        assert node.is_module_layer
        self._node = node
        self.forward_memory: List[MemoryBlock] = []
        self.backward_memory: List[MemoryBlock] = []
        self.op_used_memory: List[MemoryBlock] = []
        self.ops: List[OperatorNode] = []

    @property
    def start(self):
        return self._node.start_time

    @property
    def end(self):
        return self._node.end_time

    def memory_plot(self):
        memories = copy.deepcopy(self.forward_memory) + copy.deepcopy(
            self.backward_memory
        )
        memories = sorted(memories, key=lambda x: x.alloc_time)
        fig = go.Figure()
        y = 0
        blocks: List[Dict[str, Union[float, int]]] = []
        for mem in memories:
            block = {
                "x0": mem.alloc_time,
                "x1": mem.free_time,
                "y0": y,
                "y1": y + 0.5,
                "bytes": mem.bytes,
            }
            blocks.append(block)
            y += 0.5

        max_x = max([block["x1"] for block in blocks if block["x1"] is not None])
        min_x = min([block["x0"] for block in blocks])
        max_y = len(memories) * 0.5
        min_y = 0
        for block in blocks:
            fig.add_shape(
                type="rect",
                x0=block["x0"],
                x1=block.get("x1") or max_x,
                y0=block["y0"],
                y1=block["y1"],
                line=dict(color="black"),
                fillcolor="blue",
            )
            fig.add_annotation(
                x=(block["x0"] + (block.get("x1") or max_x)) / 2,
                y=(block["y0"] + block["y1"]) / 2,
                text=format_memory(block["bytes"]),
                showarrow=False,
                font=dict(size=12),
            )

        fig.update_layout(
            xaxis=dict(
                range=[min_x, max_x], title="Time"
            ),  # Setting the range for x-axis
            yaxis=dict(
                range=[min_y, max_y], showticklabels=False
            ),  # Setting the range for y-axis and hiding labels
            showlegend=False,
            height=max_y * 100,
            width=800,
            margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins for better visibility
        )
        fig.show()

    def non_temp_forward_memory(self) -> List[MemoryBlock]:
        if len(self.ops) == 0:
            return []
        return list(
            filter(
                lambda x: x.free_time is None or x.free_time > self.end,
                self.forward_memory,
            )
        )

    def non_temp_backward_memory(self) -> List[MemoryBlock]:
        if len(self.ops) == 0:
            return []
        back_end_time = max([mem.end_time for mem in self.ops])
        return list(
            filter(
                lambda x: x.free_time is None or x.free_time > back_end_time,
                self.backward_memory,
            )
        )


class ProfilerDataCategory(Enum):
    USER_ANNOTATION = "user_annotation"
    PYTHON_FUNCTION = "python_function"
    CPU_OP = "cpu_op"
    CPU_INSTANT_EVENT = "cpu_instant_event"
    FORWARD_BACKWARD = "fwdbwd"


class IterationData:
    def __init__(self, data: dict):
        self._data = data
        self._cat = {}
        # cpu_op events are kept as raw trace dicts; OperatorNode objects are
        # materialized lazily (and memoized) on first touch, because only a
        # small fraction of operators is ever consumed by the analysis.
        self._op_cache: Dict[int, OperatorNode] = {}
        self._cpu_ops_sorted: Optional[List] = None
        self._sequence_ops: Dict[int, List[dict]] = {}
        self._layer = None
        self._ops: Optional[List[OperatorNode]] = None
        self._memory: Optional[Dict[int, List[MemoryBlock]]] = None
        self._zero_grad: Optional[Tuple[Union[int, float], Union[int, float]]] = None
        self._optimiser_step: Optional[Tuple[Union[int, float], Union[int, float]]] = (
            None
        )
        self._optimiser: Optional[str] = None
        self._ops_search_data: Optional[Dict[Union[int, float], OperatorNode]] = None
        self._ops_search_keys: Optional[List[Union[int, float]]] = None
        self._memory_search_src: Optional[Dict[int, List[MemoryBlock]]] = None
        self._memory_search_keys: Optional[List[int]] = None

    @property
    def start(self):
        return self._data["ts"]

    @property
    def end(self):
        return self._data["ts"] + self._data["dur"]

    @property
    def optimiser_step(self) -> Optional[Tuple[Union[int, float], Union[int, float]]]:
        return self._optimiser_step

    @property
    def zero_grad_time(self) -> Optional[Tuple[Union[int, float], Union[int, float]]]:
        return self._zero_grad

    @property
    def optimiser_name(self) -> Optional[str]:
        return self._optimiser

    @property
    def cpu_ops(self) -> List[Tuple[Union[int, float], object]]:
        """(start_time, raw cpu_op event) tuples sorted by start time."""
        if self._cpu_ops_sorted is None:
            data = self._cat.get(ProfilerDataCategory.CPU_OP.value, ())
            self._cpu_ops_sorted = sorted(data, key=lambda x: x[0])
        return self._cpu_ops_sorted

    def _materialize_op(self, event) -> OperatorNode:
        node = self._op_cache.get(id(event))
        if node is None:
            node = OperatorNode(value=event_to_dict(event))
            self._op_cache[id(event)] = node
        return node

    @property
    def dataset_load_time(self) -> list[float]:
        load_times = []
        first_layer = list(self.get_layers().values())[0]
        for op in first_layer.ops:
            if op.function_name == "to":
                load_times.append(op.start_time)
        # todo: temporary fix: Due to profiling data from HuggingFace Trainer execute a pre-process stage before
        #       actually loading dataset, it leads for xMem failure, which is caused by not loading right dataset
        #       loading activities.
        #       This fix will load dataset timestamp from cpu_op event if it does not exist in Layer
        if len(load_times) == 0:
            for op in self.get_operators():
                if op.function_name == "to":
                    load_times.append(op.start_time)
        return load_times

    def add_layer(self, node: StackNode):
        if node.function_name == "zero_grad":
            if self.start <= node.start_time <= self.end:
                if self._zero_grad is None:
                    self._zero_grad = (node.start_time, node.start_time + node.duration)
                return
        # filter out the layer that belongs to the optimiser step
        # which could be fetched by the optimiser_step time
        if self.optimiser_step is not None:
            start, end = self.optimiser_step
            if start <= node.start_time:
                return
        if self.start <= node.start_time <= self.end:
            if "layer" not in self._cat.keys():
                self._cat["layer"] = {}
            name = node.function_name
            if name in self._cat["layer"].keys():
                _name = f"{name}_{uuid4().hex[:2]}"
                logger.warning(
                    f"Duplicate layer name found: {name}. Renaming to {_name}"
                )
                name = _name
            self._cat["layer"][name] = Layer(node)

    def add_event(self, event):
        timestamp = event.ts
        if timestamp <= self.end:
            cat = event.cat or "non-category"

            # Only CPU_INSTANT_EVENT events are considered before the start of the iteration
            # The memory block needs to maintain its continuity and sequence without disruption.
            if cat != ProfilerDataCategory.CPU_INSTANT_EVENT.value:
                # Thus, any event that occurs before the start of the iteration is ignored.
                if timestamp < self.start:
                    return

            # Process the Date and group them by category
            if cat not in self._cat.keys():
                self._cat[cat] = []
            if cat == ProfilerDataCategory.PYTHON_FUNCTION.value:
                node = StackNode(value=event_to_dict(event))
                start_time = node.start_time
            elif cat == ProfilerDataCategory.CPU_OP.value:
                node = event  # decoded event; materialized via _materialize_op
                start_time = timestamp
                seq_number = event_seq_number(event)
                if seq_number is not None:
                    if seq_number not in self._sequence_ops.keys():
                        self._sequence_ops[seq_number] = []
                    self._sequence_ops[seq_number].append(event)
            elif cat == ProfilerDataCategory.CPU_INSTANT_EVENT.value:
                node = CpuInstantNode(value=event_to_dict(event))
                start_time = node.start_time
            elif cat == ProfilerDataCategory.USER_ANNOTATION.value:
                pattern_zero_grad = "^Optimizer.zero_grad#[a-zA-Z0-9]+.zero_grad$"
                pattern_optimizer_step = "^Optimizer.step#[a-zA-Z0-9]+.step$"
                if re.match(pattern_zero_grad, event.name):
                    self._zero_grad = (event.ts, event.ts + event.dur)
                elif re.match(pattern_optimizer_step, event.name):
                    self._optimiser_step = (event.ts, event.ts + event.dur)
                    optimiser_name_pattern = r"#(\w+)\."
                    match = re.search(optimiser_name_pattern, event.name)
                    if match:
                        self._optimiser = match.group(1)
                return
            else:
                node = event
                start_time = timestamp
            self._cat[cat].append((start_time, node))

    def layer_summary(self) -> List[Dict[str, Any]]:
        layer_dict = []
        for name, layer in self.get_layers().items():
            if "loss" in name.lower() or "cross" in name.lower():
                forward_mem = layer.forward_memory
                backward_mem = layer.backward_memory
            else:
                forward_mem = layer.non_temp_forward_memory()
                backward_mem = layer.non_temp_backward_memory()

            layer_dict.append(
                {
                    "name": name,
                    "start": layer.start,
                    "end": layer.end,
                    "forward_memory": [memory for memory in forward_mem],
                    "backward_memory": [memory for memory in backward_mem],
                }
            )
        return layer_dict

    def optimiser_memory(self) -> List[MemoryBlock]:
        if self._optimiser_step is None:
            memory = []
        else:
            start, end = self._optimiser_step
            memory = self.memory_search(start, end)
        return memory

    def get_layers(self) -> Dict[str, Layer]:
        if self._layer is None:
            import copy

            layers = copy.deepcopy(self._cat.get("layer", {}))
            parent_layers = []
            for name, layer in tqdm(
                layers.items(), desc="Analyzer: attributing layer memory", unit="layer"
            ):
                back_trace = [
                    trace.function_name
                    for trace in layer._node.backward_stack()
                    if trace.is_module_layer
                ]
                parent_layers.extend(back_trace[1:])

                forward_memory: List[MemoryBlock] = []
                backward_memory: List[MemoryBlock] = []
                op_used_memory: List[MemoryBlock] = []
                # Add layer timerange memory blocks
                memory = self.memory_search(layer.start, layer.end)
                forward_memory.extend(memory)
                # get stackup op list, including backward and forward ops
                ops = self.ops_search(layer.start, layer.end)
                ops_with_memory = []
                for op in ops:
                    memory = self.memory_search(op.start_time, op.end_time)
                    # add all memory blocks into backward_memory, and then remove forward_memory by set.difference
                    backward_memory.extend(memory)
                    # only add memory blocks that are used by leaf ops
                    # ensuring no duplicate memory blocks or temporary memory blocks
                    for trace in op.forward_stack():
                        leaf_op = trace[-1]
                        op_memory = self.memory_search(
                            leaf_op.start_time, leaf_op.end_time
                        )
                        if len(op_memory) > 0:
                            op.memory.extend(op_memory)
                            op_used_memory.extend(op_memory)
                    op.memory = list(set(op.memory))
                    ops_with_memory.append(op)

                set_forward_memory = set(forward_memory)
                set_backward_memory = set(backward_memory).difference(forward_memory)
                set_op_used_memory = set(op_used_memory)

                layer.forward_memory.extend(set_forward_memory)
                layer.backward_memory.extend(set_backward_memory)
                layer.op_used_memory.extend(set_op_used_memory)
                layer.ops = list(set(ops_with_memory))
            for parent_layer in set(parent_layers):
                del layers[parent_layer]
            self._layer = layers
        return self._layer

    def get_operators(self) -> List[OperatorNode]:
        if self._ops is None:
            raw_ops = dict(self._cat.get(ProfilerDataCategory.CPU_OP.value, [])).values()
            data = sorted(
                (self._materialize_op(event) for event in raw_ops),
                key=lambda x: x.start_time,
            )
            ops = self._stackup_nodes(data)
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(self._op_get_memory, ops)

                for op, memory in zip(ops, results):
                    op.memory = memory
            self._ops = ops
        return self._ops

    def _op_get_memory(self, op):
        start = op.start_time
        end = op.end_time
        return self.memory_search(start, end)

    def _stackup_nodes(self, nodes: List[ProfilerNode]) -> List[ProfilerNode]:
        # Nodes sorted by (start, -end) are visited outer-before-inner, so a
        # stack of currently open intervals yields each node's innermost
        # container in O(n log n) overall.
        intervals = [(node.start_time, node.end_time) for node in nodes]
        groups: Dict[Tuple[int, int], List[ProfilerNode]] = {}
        for node, interval in zip(nodes, intervals):
            groups.setdefault(interval, []).append(node)
        order = sorted(
            range(len(nodes)), key=lambda i: (intervals[i][0], -intervals[i][1], i)
        )
        root_stacks: List[ProfilerNode] = []
        stack: List[Tuple[int, int, ProfilerNode]] = []
        for i in order:
            node = nodes[i]
            start, end = intervals[i]
            while stack and stack[-1][1] < end:
                stack.pop()
            if len(groups[(start, end)]) > 1:
                pass  # identical-interval twins are wired after the sweep
            elif stack:
                stack[-1][2].add_child(node)
            else:
                root_stacks.append(node)
            stack.append((start, end, node))
        # Nodes sharing an identical (start, end) interval mutually contain
        # each other: every earlier twin becomes a child of the last one and
        # the last a child of the second-to-last, so none of them is a root.
        # This mirrors the previous O(n^2) implementation exactly.
        for group in groups.values():
            if len(group) > 1:
                for twin in group[:-1]:
                    group[-1].add_child(twin)
                group[-2].add_child(group[-1])
        return root_stacks

    def get_memory_activities(self) -> Dict[int, List[MemoryBlock]]:
        if self._memory is None:
            _address: Dict[str, MemoryBlock] = {}
            _activities: Dict[int, List[MemoryBlock]] = {}
            _data = self._cat[ProfilerDataCategory.CPU_INSTANT_EVENT.value]
            _data = sorted(_data, key=lambda x: x[0])
            for trace in _data:
                _node = trace[1]
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
            _activities = dict(sorted(_activities.items()))
            self._memory = _activities
        return self._memory

    def _search_trace(
        self,
        list_data: Dict[int, List[Any]],
        start: Optional[Union[float, int]],
        end: Optional[Union[float, int]],
        sorted_timestamps: Optional[list] = None,
    ) -> list:
        start = start or 0
        end = end or max(list_data.keys())
        if sorted_timestamps is None:
            sorted_timestamps = list(list_data.keys())
        start_idx = bisect_left(sorted_timestamps, start)
        end_idx = bisect_right(sorted_timestamps, end)

        result = []
        for i in range(start_idx, end_idx):
            events = list_data[sorted_timestamps[i]]
            if isinstance(events, list):
                result.extend(events)
            else:
                result.append(events)
        return result

    def memory_search(
        self, start: Optional[float] = None, end: Optional[float] = None
    ) -> List[MemoryBlock]:
        # The activities dict can be swapped by construct_memory_sequence, so
        # the cached key list is tied to the dict's identity.
        activities = self.get_memory_activities()
        if self._memory_search_src is not activities:
            self._memory_search_src = activities
            self._memory_search_keys = list(activities.keys())
        # Activities are keyed by alloc_time and the key list is sorted, so
        # _search_trace already returns blocks in alloc_time order.
        return self._search_trace(
            activities, start, end, self._memory_search_keys
        )

    def ops_search(
        self, start: Optional[float] = None, end: Optional[float] = None
    ) -> List[OperatorNode]:
        if self._ops_search_data is None:
            self._ops_search_data = dict(
                self._cat[ProfilerDataCategory.CPU_OP.value]
            )
            self._ops_search_keys = list(self._ops_search_data.keys())
        result = self._search_trace(
            self._ops_search_data, start, end, self._ops_search_keys
        )
        sequence_ids = set(
            [
                seq_number
                for seq_number in map(event_seq_number, result)
                if seq_number is not None
            ]
        )
        for _id in sequence_ids:
            result.extend(self._sequence_ops.get(_id, []))
        # raw events are deduplicated by identity (as the node objects were)
        # and only the touched subset is materialized into OperatorNodes
        deduped = {id(event): event for event in result}.values()
        result = self._stackup_nodes(
            [self._materialize_op(event) for event in deduped]
        )
        return sorted(result, key=lambda x: x.start_time)


class ProfilerDataProcessing:
    def __init__(self, file_path: str):
        self._file_path = file_path
        self._time_based_data = {}
        self._iteration: Dict[int, IterationData] = {}
        self._python_stack: Dict[int, StackNode] = {}
        self._zero_grad: Optional[int] = None
        self._optimiser_step: Optional[Tuple[int, int]] = None
        self._load()

    @property
    def max_iterations(self):
        return len(self._iteration)

    def get_iteration(self, iteration: int) -> IterationData:
        assert iteration > 0
        return self._iteration.get(iteration - 1, None)

    def load_data(self) -> List:
        """Decode the trace into event objects with attribute access.

        With msgspec available, events are decoded against a schema and the
        heavyweight `args` payloads stay as raw JSON bytes until touched.
        Without it, stdlib json is used and dicts are wrapped for the same
        attribute interface.
        """
        _t0 = time.perf_counter()
        if msgspec is not None:
            with open(self._file_path, "rb") as file:
                events = _TRACE_DECODER.decode(file.read()).traceEvents
        else:
            with open(self._file_path, "r") as file:
                _data = json.load(file)
            events = [
                SimpleNamespace(
                    name=e.get("name", ""),
                    cat=e.get("cat", ""),
                    ph=e.get("ph", ""),
                    ts=e.get("ts", 0),
                    dur=e.get("dur", 0),
                    pid=e.get("pid", 0),
                    tid=e.get("tid", 0),
                    args=e.get("args", {}),
                )
                for e in _data.get("traceEvents", ())
            ]
        logger.info(
            "Analyzer: parsed %s events in %.2fs",
            len(events), time.perf_counter() - _t0,
        )
        return events

    def _load(self):
        events = self.load_data()
        _t0 = time.perf_counter()
        iteration_count = 0
        iteration_windows = []
        for element in tqdm(
            events, desc="Analyzer: processing trace events", unit="ev"
        ):
            name = element.name
            cat = element.cat

            if cat == "user_annotation":
                pattern = "^ProfilerStep#[0-9]+$"
                if re.match(pattern, name):
                    self._iteration[iteration_count] = IterationData(
                        data=event_to_dict(element)
                    )
                    iteration_count += 1
                    iteration_windows = [
                        (it.start, it.end, it) for it in self._iteration.values()
                    ]
                else:
                    for iteration in self._iteration.values():
                        iteration.add_event(element)
            elif cat == "python_function":
                stack = StackNode(value=event_to_dict(element))
                if stack.parent_id in self._python_stack.keys():
                    self._python_stack[stack.parent_id].add_child(stack)
                self._python_stack[stack.id] = stack
                if stack.is_module_layer or stack.function_name == "zero_grad":
                    for iteration in self._iteration.values():
                        iteration.add_layer(stack)
            elif cat == "cpu_instant_event":
                # instant events belong to every iteration ending after them
                for _start, _end, iteration in iteration_windows:
                    if element.ts <= _end:
                        iteration.add_event(element)
            else:
                # dispatching by window here avoids offering every event to
                # every iteration just to be rejected inside add_event; no
                # break, in case windows share an exact boundary timestamp
                _ts = element.ts
                for _start, _end, iteration in iteration_windows:
                    if _start <= _ts <= _end:
                        iteration.add_event(element)

        sorted(self._time_based_data.keys())
        logger.info(
            "Analyzer: event processing took %.2fs", time.perf_counter() - _t0
        )
