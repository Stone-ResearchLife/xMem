import json
import multiprocessing
import re
import copy
import logging
from uuid import uuid4
import plotly.graph_objects as go
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Tuple
from bisect import bisect_left, bisect_right
from perf_estimator.utilis.utilis import format_memory
from . import StackNode, OperatorNode, CpuInstantNode, MemoryBlock, ProfilerNode


logger = logging.getLogger(__name__)


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
        self._sequence_ops: Dict[int, List[OperatorNode]] = {}
        self._layer = None
        self._ops: Optional[List[OperatorNode]] = None
        self._memory: Optional[Dict[int, List[MemoryBlock]]] = None
        self._zero_grad: Optional[Tuple[Union[int, float], Union[int, float]]] = None
        self._optimiser_step: Optional[Tuple[Union[int, float], Union[int, float]]] = (
            None
        )
        self._optimiser: Optional[str] = None

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
    def cpu_ops(self) -> List[Tuple[Union[int, float], OperatorNode]]:
        data = self._cat.get(ProfilerDataCategory.CPU_OP.value, ())
        data = sorted(data, key=lambda x: x[0])
        return data

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

    def add_event(self, event: dict):
        timestamp = event["ts"]
        if timestamp <= self.end:
            cat = event.get("cat", "non-category")

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
                node = StackNode(value=event)
                start_time = node.start_time
            elif cat == ProfilerDataCategory.CPU_OP.value:
                node = OperatorNode(value=event)
                start_time = node.start_time
                if node.seq_number is not None:
                    if node.seq_number not in self._sequence_ops.keys():
                        self._sequence_ops[node.seq_number] = []
                    self._sequence_ops[node.seq_number].append(node)
            elif cat == ProfilerDataCategory.CPU_INSTANT_EVENT.value:
                node = CpuInstantNode(value=event)
                start_time = node.start_time
            elif cat == ProfilerDataCategory.USER_ANNOTATION.value:
                pattern_zero_grad = "^Optimizer.zero_grad#[a-zA-Z0-9]+.zero_grad$"
                pattern_optimizer_step = "^Optimizer.step#[a-zA-Z0-9]+.step$"
                if re.match(pattern_zero_grad, event["name"]):
                    self._zero_grad = (event["ts"], event["ts"] + event["dur"])
                elif re.match(pattern_optimizer_step, event["name"]):
                    self._optimiser_step = (event["ts"], event["ts"] + event["dur"])
                    optimiser_name_pattern = r"#(\w+)\."
                    match = re.search(optimiser_name_pattern, event["name"])
                    if match:
                        self._optimiser = match.group(1)
                return
            else:
                node = event
                start_time = node["ts"]
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
            for name, layer in layers.items():
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
            data = dict(self._cat.get(ProfilerDataCategory.CPU_OP.value, [])).values()
            data = sorted(data, key=lambda x: x.start_time)
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
        root_stacks: List[ProfilerNode] = []
        for node in nodes:
            matched_result = list(
                filter(
                    lambda x: x.start_time <= node.start_time
                    and x.end_time >= node.end_time,
                    nodes,
                )
            )
            matched_result.remove(node)
            if len(matched_result) == 0:
                root_stacks.append(node)
            else:
                matched_result.sort(key=lambda x: (x.start_time, -x.end_time))
                matched_result[-1].add_child(node)
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
    ) -> list:
        start = start or 0
        end = end or max(list_data.keys())
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
        result = self._search_trace(self.get_memory_activities(), start, end)
        return sorted(result, key=lambda x: x.alloc_time)

    def ops_search(
        self, start: Optional[float] = None, end: Optional[float] = None
    ) -> List[OperatorNode]:
        result = self._search_trace(
            dict(self._cat[ProfilerDataCategory.CPU_OP.value]), start, end
        )
        sequence_ids = set(
            [op.seq_number for op in result if op.seq_number is not None]
        )
        for _id in sequence_ids:
            result.extend(self._sequence_ops.get(_id, []))
        result = self._stackup_nodes(list(set(result)))
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

    def load_data(self) -> dict:
        with open(self._file_path, "r") as file:
            _data = json.load(file)
        return _data

    def _load(self):
        events = self.load_data()
        iteration_count = 0
        for element in events["traceEvents"]:
            name = element.get("name", "")
            cat = element.get("cat", "")

            if cat == "user_annotation":
                pattern = "^ProfilerStep#[0-9]+$"
                if re.match(pattern, name):
                    self._iteration[iteration_count] = IterationData(data=element)
                    iteration_count += 1
                else:
                    for iteration in self._iteration.values():
                        iteration.add_event(element)
            elif cat == "python_function":
                stack = StackNode(value=element)
                if stack.parent_id in self._python_stack.keys():
                    self._python_stack[stack.parent_id].add_child(stack)
                self._python_stack[stack.id] = stack
                if stack.is_module_layer or stack.function_name == "zero_grad":
                    for iteration in self._iteration.values():
                        iteration.add_layer(stack)
            else:
                for iteration in self._iteration.values():
                    iteration.add_event(element)

        sorted(self._time_based_data.keys())
