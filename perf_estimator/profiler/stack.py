from typing import Any, Dict, List, Optional, Tuple
from hashlib import sha256
from .node import StackNode


class StackLeaf:
    def __init__(self, node: StackNode):
        assert isinstance(node, StackNode)
        self._leaf = node
        self._hierarchy = None

    @property
    def leaf(self) -> StackNode:
        return self._leaf

    @property
    def hierarchy(self) -> List[StackNode]:
        return self._get_model_layer_trace()

    @property
    def is_model_layer_trace(self) -> bool:
        self._hierarchy = self._get_model_layer_trace()
        return len(self._hierarchy) > 0

    @property
    def module_name(self) -> str:
        """
        Generate a module name for the stack leaf.
        The module name will be the function name of the leaf if the leaf is not a module layer.
        The last module lauer name will be extracted from the backward stack if the leaf is a module layer.

        Returns:

        """
        _hierarchy = self._get_model_layer_trace()
        if len(_hierarchy) == 0:
            _hierarchy = [self.leaf]
        return "->".join([node.function_name for node in _hierarchy])

    @property
    def leaf_id(self) -> str:
        return sha256(self.flame_string().encode()).hexdigest()

    def __repr__(self):
        return f"{self.module_name}(Layer Trace: {self.is_model_layer_trace}): start: {self.leaf.start_time}, dur: {self.leaf.duration}"

    def _get_model_layer_trace(self) -> List[StackNode]:
        module_hierarchy = self._hierarchy
        if module_hierarchy is None:
            module_hierarchy = [
                node.parent
                for node in self.leaf.backward_stack()
                if node.is_module_layer
            ]
            module_hierarchy.reverse()
        return module_hierarchy

    def flame_string(
        self, weight: Optional[int] = None, module_only: bool = False
    ) -> Optional[str]:
        """Generate a flame graph string for the stack leaf.

        Args:
            weight (int, optional): weight is used to calculate the width of the flame graph. Defaults to None.
                                    it can be memory, time, or any other metric.
            module_only (bool): if True, only module layers are included in the flame graph.

        Returns:
            str: flame graph string, the string is in the format of "trace1;trace2;trace3;...;traceN <weight>"
        """
        trace_list = []
        if module_only:
            trace_list = self._get_model_layer_trace()
        else:
            for node in self.leaf.backward_stack():
                trace_list.append(node)
            trace_list = list(reversed(trace_list))
        if len(trace_list) == 0:
            return None
        return (
            "; ".join([node.name for node in trace_list])
            + f" {weight if weight else 1}"
        )

    def to_json(self) -> Dict[str, Any]:
        """reformat the stack leaf to a json format.
        The field `id` is used to identify if the stack leaf is the same as another stack leaf
        based on the flame graph string.

        Returns:
            Dict[str, Any]: a dictionary contains the information of the stack leaf.

        """
        return {
            "layer": self.module_name,
            "start": self.leaf.start_time,
            "end": self.leaf.end_time,
            "duration": self.leaf.duration,
            "id": self.leaf_id,
        }
