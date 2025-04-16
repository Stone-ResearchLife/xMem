import torch
from typing import List, Tuple, Dict, Optional


class ModelInfo:
    def __init__(self, model: torch.nn.Module):
        assert isinstance(model, torch.nn.Module)
        self._model: torch.nn.Module = model
        self._layer_stack: List[List[Tuple[str, torch.nn.Module]]] = []
        self._dfs(self._model, [], self._layer_stack)

    @property
    def name(self) -> str:
        return self._model.__class__.__name__

    @property
    def layer_stack(self) -> List[List[Tuple[str, torch.nn.Module]]]:
        return self._layer_stack

    def id_to_layerstack_map(
        self,
    ) -> Dict[int, Dict[str, List[Tuple[str, torch.nn.Module]]]]:
        _id_map = {}
        for _layer_stack in self._layer_stack:
            _id = id(_layer_stack[-1][1])
            _stack_names = [_layer[0] for _layer in _layer_stack]
            _id_map[_id] = {"id": "->".join(_stack_names), "stack": _layer_stack}
        return _id_map

    def _dfs(
        self,
        node: torch.nn.Module,
        current_path: list,
        all_paths: list,
        layer_index=None,
    ):
        if layer_index is None:
            layer_index = {}
        assert isinstance(node, torch.nn.Module)
        _name = node.__class__.__name__
        if _name not in layer_index.keys():
            layer_index[_name] = 0
        else:
            layer_index[_name] += 1
        _key = f"{_name}_{layer_index[_name]}"
        current_path.append((_key, node))

        _modules = list(node._modules.values())
        if len(_modules) == 0:
            all_paths.append(list(current_path))
        else:
            for sub_module in _modules:
                self._dfs(sub_module, current_path, all_paths, layer_index)
        # pop up the last index of the current path, and go back to the parent node
        # Example:
        # current_path = [1, 2, 3]
        # after pop: [1, 2]
        current_path.pop()
