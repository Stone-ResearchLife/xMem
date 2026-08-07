from __future__ import annotations
import re
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from perf_estimator.data_structure import TreeNode
from functools import cached_property


class ProfilerNode(TreeNode, ABC):
    def __init__(self, value: dict):
        assert isinstance(value, dict)
        super().__init__(value)
        self.__parsed_name: Optional[list] = None

    @property
    def ph(self) -> str:
        return self.value["ph"]

    @property
    def category(self):
        return self.value.get("cat", None)

    @property
    def name(self) -> str:
        return self.value["name"]

    @property
    def _parsed_name_cached(self) -> list:
        if self.__parsed_name is None:
            self.__parsed_name = self._parse_name()
        return self.__parsed_name

    @property
    def namespace_name(self) -> str:
        return str(self._parsed_name_cached[0]).strip()

    @property
    def function_name(self) -> str:
        return " ".join(self._parsed_name_cached[1:]).strip()

    @property
    def start_time(self) -> int:
        return self.value["ts"]

    @cached_property
    def duration(self) -> int:
        return self.value.get("dur", 0)

    @property
    def end_time(self) -> int:
        return self.start_time + self.duration

    @property
    def tid(self) -> int:
        return self.value["tid"]

    @property
    def pid(self) -> int:
        return self.value["pid"]

    def __repr__(self):
        return f"{self.namespace_name}::{self.function_name}({self.category}): {self.start_time}->{self.end_time}"

    @abstractmethod
    def _parse_name(self) -> list:
        """Parse the name of the node into two parts: namespace and function name
        The return value is a list of two or more elements, the first element is the namespace, and the rest of element is the function name

        Examples:
            The name of the node is "threading.py(1016): _bootstrap_inner"
            The function will return a list of two elements, the first element is "threading.py", and the second element is "_bootstrap_inner"

        """
        pass


class StackNode(ProfilerNode):
    """The class represents a node in the stack tree of the profiler data.

    Examples:
        The original data structure is a dictionary:
            {
                "ph": "X",
                "cat": "python_function",
                "name": "threading.py(1016): _bootstrap_inner",
                "pid": 0,
                "tid": 0,
                "ts": 1724696154696855,
                "dur": 416124,
                "args": {
                    "Python parent id": 1,
                    "Python id": 2,
                    "Ev Idx": 425
                }
            }
    """

    def __init__(self, value: dict):
        """Build a node for each paython function call

        Args:
            value (dict): The dictionary of the node
        """
        assert value["cat"] == "python_function"
        super().__init__(value)

    @property
    def id(self) -> int:
        return self.value["args"]["Python id"]

    @property
    def parent_id(self) -> Optional[int]:
        return self.value["args"]["Python parent id"]

    @property
    def is_module_layer(self) -> bool:
        """
        The internal function _call_impl is generally called after the module layer function.
        Thus, the function can be used to determine whether the parent node exhibits the module layer name

        Examples:
            this layer's name is "torch/nn/modules/module.py(1534): _call_impl"
            parent layer's name is "nn.Module: ReLU_0"

        It can help us rapidly locate the module layer in the stack tree.
        """
        # return "_call_impl" in self.function_name
        return (
            "nn.Module" in self.namespace_name
            or "transformers/loss/loss_utils.py" in self.namespace_name
        )

    @property
    def is_getattr(self) -> bool:
        """Determine whether the function is the __getattr__ function"""
        return "__getattr__" in self.function_name

    def __repr__(self):
        return f"StackNode(children:{len(self.children)}): {self.name}, start: {self.start_time}, end: {self.end_time}"

    def _parse_name(self) -> List[str]:
        """Parse the python function name into two parts: file name and function name

        Examples:
            There are two types of function name:
                1. threading.py(1016): _bootstrap_inner
                2. <built-in method _abc._abc_subclasshook>
            The first type is the normal function call, and the second type is the built-in function call.
            The function will return a list of two/more elements, the first element is the file name, and the second element is the function name.

        Returns:
            list: The list of the file name and function name
        """
        pattern = r"<built-in\s+(method|function)\s+\w+.*?>"
        matcher = re.match(pattern, self.name)
        if matcher:
            pattern = r"<built-in\s+(method|function)"
            split_text = re.split(pattern, self.name)
            split_text.pop(0)
            split_text[0] = "built-in " + split_text[0]
        else:
            split_text = self.name.split(":")
        if len(split_text) == 1:
            split_text.append("unknown")
        return split_text


class OperatorNode(ProfilerNode):
    """
    This node represents one of cpu_op type events in the profiler data.

    Examples:
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "aten::clamp_min",
            "pid": 6632,
            "tid": 6632,
            "ts": 1724696154808890,
            "dur": 1995,
            "args": {
                "External id": 26,
                "Concrete Inputs": ["", "0"],
                "Input type": ["float", "Scalar"],
                "Input Dims": [[200, 32, 43, 43], []],
                "Ev Idx": 25,
                "Sequence number": 22, // optional
                "Fwd thread id": 0 // optional
            }
        },
    """

    def __init__(self, value: Dict[str, Any]):
        assert value["cat"] in ["cpu_op"]
        super().__init__(value)
        self.memory = []

    @property
    def args_number(self) -> int:
        return len(self.input_types)

    @property
    def input_types(self) -> List[str]:
        return self.value["args"].get("Input type", [])

    @property
    def input_args(self) -> List[Dict[str, Any]]:
        _args = []
        _types = self.input_types
        for index, arg in enumerate(self.value["args"].get("Input Dims", [])):
            if len(arg) > 0:
                bytes = np.prod(arg) * 4
                _args.append(
                    {"type": _types[index], "dims": arg, "index": index, "bytes": bytes}
                )
        return _args

    @property
    def concrete_inputs(self) -> List[Dict[str, Any]]:
        _concrete_inputs = []
        _type = self.input_types
        for index, arg in enumerate(self.value["args"].get("Concrete Inputs", [])):
            if len(arg) > 0:
                _concrete_inputs.append(
                    {"concrete_input": arg, "type": _type[index], "index": index}
                )
        return _concrete_inputs

    @property
    def seq_number(self) -> Optional[int]:
        return self.value["args"].get("Sequence number", None)

    @property
    def forward_thread_id(self) -> Optional[int]:
        return self.value["args"].get("Fwd thread id", None)

    @property
    def is_autograd_enabled(self) -> bool:
        return self.seq_number is not None

    @property
    def is_aten_op(self) -> bool:
        return "aten" in self.name

    def __repr__(self):
        return f"OperatorNode(seq:{self.seq_number}): {self.name}, start: {self.start_time}, end: {self.end_time}"

    def _parse_name(self) -> list:
        if ": " in self.name:
            return self.name.split(": ")
        elif "::" in self.name:
            return self.name.split("::")
        else:
            return ["op", self.name]


class CpuInstantNode(ProfilerNode):
    def __init__(self, value: dict):
        assert value["cat"] == "cpu_instant_event"
        super().__init__(value)

    @property
    def _args(self) -> dict:
        return self.value["args"]

    @property
    def total_reserved(self) -> int:
        return self._args["Total Reserved"]

    @property
    def total_allocated(self) -> int:
        return self._args["Total Allocated"]

    @property
    def bytes(self) -> int:
        return self._args["Bytes"]

    @property
    def address(self) -> int:
        return self._args["Addr"]

    @property
    def device_id(self) -> int:
        return self._args["Device Id"]

    @property
    def device_type(self) -> int:
        return self._args["Device Type"]

    def _parse_name(self) -> list:
        return ["cpu", "memory"]
