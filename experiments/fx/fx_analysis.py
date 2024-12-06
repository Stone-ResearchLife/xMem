import torch
import copy
from typing import List, Optional, Dict, Any, Tuple, Union
from collections import OrderedDict
from .model_info import ModelInfo


class EnhancedGraphNode:
    def __init__(self, op: torch.fx.Node):
        self._op: torch.fx.Node = op
        self._layer_module: Optional[torch.nn.Module] = None
        self._input : Optional[List[EnhancedGraphNode]] = None
        self._output: Optional[torch.Tensor] = None
        self._weight: Optional[torch.nn.parameter.Parameter] = None
        self._bias: Optional[torch.nn.parameter.Parameter] = None
        self._stack: Optional[Dict[str, Union[str, List[Tuple[str, torch.nn.Module]]]]] = None
        self._gradients = []

    @property
    def index(self) -> str:
        stacks = copy.deepcopy(self.stack_modules)
        if stacks is None:
            return "0"

        if len(stacks) == 1:
            split_names = self.name.split("_")
            if len(split_names) >= 1 and split_names[-1].isdigit():
                return split_names[-1]
        else:
            index_string = ""
            for stack in stacks:
                stack_name = stack[0]
                stack_index = stack_name.split("_")[-1]
                if stack_index.isdigit():
                    index_string += stack_index
                else:
                    index_string += "0"
            return index_string

    @property
    def name(self) -> str:
        return self._op.name

    @property
    def real_layer_name(self):
        stacks = copy.deepcopy(self.stack_modules)
        if stacks is not None:
            index_string = ""
            for stack in stacks:
                stack_name = stack[0]
                stack_index = stack_name.split("_")[-1]
                if stack_index.isdigit():
                    index_string += stack_index
                else:
                    index_string += "0"
            layer_name = str(self.stack_modules[-1][0]).split("_")
            if len(layer_name) > 1 and layer_name[-1].isdigit():
                layer_name = "_".join(layer_name[:-1])
            return f"{layer_name}_{index_string}"
        return self.name

    @property
    def backward_name(self) -> Optional[str]:
        if self.gradient_function is not None:
            return self.gradient_function.__class__.__name__
        return None

    @property
    def backward_name_with_index(self):
        return f"{self.backward_name}_{self.index}"

    @property
    def gradient_function(self):
        if self.output is not None and self.output.grad_fn is not None:
            _name = self.output.grad_fn.__class__.__name__
            if "relu" in _name.lower() and 'relu' not in self.real_layer_name.lower():
                return self.output.grad_fn.next_functions[0][0]
            return self.output.grad_fn
        return None

    @property
    def module(self) -> torch.nn.Module:
        return self._layer_module

    @module.setter
    def module(self, value):
        assert isinstance(value, torch.nn.Module)
        self._layer_module = value

    @property
    def op_type(self) -> str:
        return self._op.op

    @property
    def traced_node(self) -> torch.fx.Node:
        return self._op

    @property
    def model(self) -> torch.fx.GraphModule:
        return torch.fx.symbolic_trace(self._op.graph.owning_module)

    @property
    def all_input_nodes(self) -> List[torch.fx.Node]:
        return self._op.all_input_nodes

    @property
    def inputs(self) -> Optional[List['EnhancedGraphNode']]:
        return self._input if self._input is not None else []

    @property
    def output(self) -> Optional[torch.Tensor]:
        return self._output if self._output is not None else None

    @property
    def weight(self) -> Optional[torch.nn.parameter.Parameter]:
        return self._weight

    @property
    def bias(self) -> Optional[torch.nn.parameter.Parameter]:
        return self._bias

    @property
    def is_inplace(self):
        return hasattr(self.module, 'inplace')

    @is_inplace.setter
    def is_inplace(self, value):
        pass

    @property
    def stack_string(self) -> str:
        return self._stack.get("id", None) if self._stack is not None else None

    @property
    def stack_modules(self) -> Optional[List[Tuple[str, torch.nn.Module]]]:
        return self._stack.get("stack", None) if self._stack is not None else None

    def add_input(self, tensor: 'EnhancedGraphNode'):
        assert isinstance(tensor, EnhancedGraphNode)
        if self._input is None:
            self._input = []
        self._input.append(tensor)

    def remove_input(self, tensor: 'EnhancedGraphNode'):
        assert isinstance(tensor, EnhancedGraphNode)
        self._input.remove(tensor)

    def set_output(self, tensor: Union[torch.Tensor, int]):
        if isinstance(tensor, int):
            tensor = torch.tensor(tensor)
        assert isinstance(tensor, torch.Tensor)
        self._output = tensor

    def set_weight(self, tensor: torch.nn.parameter.Parameter):
        assert isinstance(tensor, torch.nn.parameter.Parameter)
        self._weight = tensor

    def set_bias(self, tensor: torch.nn.parameter.Parameter):
        assert isinstance(tensor, torch.nn.parameter.Parameter)
        self._bias = tensor

    def set_stack(self, stack: Dict[str, List[Tuple[str, torch.nn.Module]]]):
        assert isinstance(stack, dict)
        self._stack = stack

    def gradient(self) -> Union[None, Dict[str, List[Dict[str, Any]]]]:
        if self.name == "output":
            return None
        if self.gradient_function is not None:
            _inputs_dict = {_input.backward_name: _input for _input in self.inputs}
            operator_grads = []
            parameter_grads = []
            self._dfs(
                node=self.gradient_function,
                current_path=[],
                parameter_grads=parameter_grads,
                operator_grads=operator_grads,
                operator_names=_inputs_dict
            )
            backward_outputs = {
                "output": operator_grads,
                "parameters": parameter_grads
            }
            return backward_outputs
        return None

    def get_saved_tensors(self) -> List[torch.Tensor]:
        _grad_fn = self.gradient()
        if _grad_fn is not None:
            gards = []
            for _outputs in _grad_fn.get("output", []):
                for _fn in _outputs[:-1]:
                    gards.append(_fn)
            for _params in _grad_fn.get("parameters", []):
                for _fn in _params:
                    gards.append(_fn)
            grads = list(set(gards))
            saved_tensors = []
            for grad in grads:
                saved_gard_tensors = [getattr(grad, grad_fn_attr) for grad_fn_attr in dir(grad) if grad_fn_attr.startswith('_saved_') and isinstance(getattr(grad, grad_fn_attr), Union[torch.Tensor, torch.nn.Parameter])]
                saved_tensors.extend(saved_gard_tensors)
            # Remove duplicate tensors. Especially for the case of Loss
            unique_tensors = []
            for t in saved_tensors:
                if not any(torch.equal(t, ut) for ut in unique_tensors):
                    unique_tensors.append(t)
            saved_tensors = unique_tensors
            if None in saved_tensors:
                saved_tensors.remove(None)
            return saved_tensors
        return []

    def to_json(self):
        _input = self._input if self._input is not None else []
        # _gradients = self.gradient()
        # _grad_output = []
        # _grad_parameters = []
        # if isinstance(_gradients, dict):
        #     for _grad in _gradients.get("output", []):
        #         _grad_output.append(_grad["shape"])
        #     for _grad in _gradients.get("parameters", []):
        #         _grad_parameters.append(_grad["shape"])

        return {
            "type": self.op_type,
            "name": self.name,
            "backward_name": self.backward_name,
            "stack": self.stack_string,
            "input":  [_tensor.output.shape for _tensor in _input],
            "output": self.output.shape if self.output is not None else None,
            "weight": self.weight.shape if self._weight is not None else None,
            "bias": self.bias.shape if self._bias is not None else None,
            "is_inplace": self.is_inplace,
            # "gradients_output": _grad_output,
            # "gradients_parameters": _grad_parameters
        }

    def dnnmem_format(self):
        # todo: remove this method late
        _input = self._input if self._input is not None else []
        _gradients = self.gradient()
        _grad_output = []
        _grad_parameters = []
        if isinstance(_gradients, dict):
            for _grad in _gradients.get("output", []):
                _grad_output.append(_grad["tensor"])
            for _grad in _gradients.get("parameters", []):
                _grad_parameters.append(_grad["tensor"])

        weight = self.weight if self._weight is not None else None
        bias = self.bias if self._bias is not None else None
        forward_input = [_tensor.output for _tensor in _input]
        forward_output = self.output if self.output is not None else None

        return {
            "initial": {
                "name": self.name,
                "parameters": {
                    "value": [weight, bias],
                    "call_index": [],
                },
            },
            "forward": {
                "name": self.name,
                "input": {
                    "value": forward_input,
                    "call_index": []
                },
                "output": {
                    "value": forward_output,
                    "call_index": [],
                },
                "parameters": {
                    "value": [weight, bias],
                    "call_index": []
                },
                "ephemeral": {
                    "value": [],
                    "call_index": []
                },
            },
            "backward": {
                "name": self.backward_name,
                "output": {
                    "value": _grad_output,
                    "call_index": []
                },
                "parameters": {
                    "value": _grad_parameters,
                    "call_index": []
                },
                "ephemeral": {
                    "value": [],
                    "call_index": []
                }
            },
        }

    def __str__(self):
        return self._build_display_string()

    def __repr__(self):
        return self._build_display_string()

    def _build_display_string(self):
        _name = self.name
        if self.stack_string is not None:
            _name = self.stack_string
        _output = self.output.shape if self.output is not None else None
        _weight = self.weight.shape if self.weight is not None else None
        _bias = self.bias.shape if self.bias is not None else None
        return f"{_name}({self.op_type}): output={_output}, weight={_weight}, bias={_bias}"

    def _dfs(self, node, current_path: list, parameter_grads: list, operator_grads: list, operator_names: dict):
        if node is None:
            return

        _op_name = node.__class__.__name__
        _key = node
        current_path.append(_key)
        if _op_name in operator_names.keys():
            operator_grads.append(list(current_path))
        elif not hasattr(node, 'next_functions') or len(list(node.next_functions)) == 0:
            parameter_grads.append(list(current_path))
        else:
            next_functions = [func[0] for func in list(node.next_functions)]
            for child in next_functions:
                self._dfs(child, current_path, parameter_grads, operator_grads, operator_names)
        # pop up the last index of the current path, and go back to the parent node
        # Example:
        # current_path = [1, 2, 3]
        # after pop: [1, 2]
        current_path.pop()


class _AnalysisInterpreter(torch.fx.Interpreter):
    def __init__(
            self,
            module: torch.nn.Module,
            garbage_collect_values: bool = True,
            graph: Optional[torch.fx.Graph] = None,
            id2layer: Optional[Dict[int, Any]] = None
    ):
        super().__init__(module, garbage_collect_values, graph)
        self._enhance_nodes: OrderedDict[str, EnhancedGraphNode] = OrderedDict()
        self._current_node: Optional[EnhancedGraphNode] = None
        self._id2layer = {}
        if id2layer is not None:
            self._id2layer = id2layer

    @property
    def op_nodes(self) -> OrderedDict[str, EnhancedGraphNode]:
        return self._enhance_nodes

    def run_node(self, node):
        _node = EnhancedGraphNode(node)
        self._current_node = _node
        result = super().run_node(node)
        # Supplement other information
        self._supplement_input(_node)
        _node.set_output(result)
        self._enhance_nodes[_node.name] = _node
        return result

    def _supplement_input(self, node: EnhancedGraphNode):
        op_names = {op.traced_node.name: op for op in self._enhance_nodes.values()}
        for input_node in node.all_input_nodes:
            _input_name = input_node.name
            if _input_name in op_names.keys():
                _input_node = op_names[_input_name]
                if _input_node.op_type in ["call_module", "call_function", "placeholder", "call_method"]:
                    node.add_input(_input_node)


    def call_module(self, target : 'Target', args, kwargs : Dict[str, Any]) -> Any:
        _module = self.fetch_attr(target)
        if isinstance(_module, torch.nn.Module):
            self._current_node.module = _module
            _stack = self._id2layer.get(id(_module), None)
            if _stack is not None:
                self._current_node.set_stack(_stack)
            for name, params in _module.named_parameters():
                if params is None:
                    continue
                if "weight" in name:
                    self._current_node.set_weight(params)
                elif "bias" in name:
                    self._current_node.set_bias(params)
        return super().call_module(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        # print("function:" + str(target))
        return super().call_function(target, args, kwargs)

    def call_method(self, target, args, kwargs) -> Any:
        # print("method:" + target)
        return super().call_method(target, args, kwargs)


class FXAnalyser:
    def __init__(
            self, model: Union[torch.nn.Module],
            data_x: torch.Tensor,
            data_y: Optional[torch.Tensor] = None,
            is_transformer: bool = False,
            loss: Optional[torch.nn.Module] = None,

    ):
        self._is_transformer = is_transformer
        self._model: Union[torch.nn.Module] = model
        self._input_tensor: torch.Tensor = data_x
        self._label_tensor: Optional[torch.Tensor] = data_y
        self._loss = loss
        self._layer_info: Optional[OrderedDict[str, EnhancedGraphNode]] = OrderedDict()
        self._dry_run()

    @property
    def input(self) -> torch.Tensor:
        return self._input_tensor

    @input.setter
    def input(self, value: torch.Tensor):
        self._input_tensor = value

    @property
    def layers_info(self) -> OrderedDict[str, EnhancedGraphNode]:
        """Returns the layer information in the model

        Examples:
        >>> analyser = FXAnalyser(model, input_tensor)
        >>> layer_info = analyser.layers_info
        >>> for name, value in layer_info.items():
        >>>     print(f"Layer: {name}")
        >>>     print(f"Node: {value}")
        >>> # Output
        >>> # Layer: conv1
        >>> # Node: Conv2d(call_module): output=[1, 64, 112, 112], weight=[64, 3, 7, 7], bias=[64]

        Returns:
            OrderedDict[str, EnhancedGraphNode]: Layer information

        """
        return self._layer_info

    def _build_layer_call_stack(self) -> Dict[int, Dict[str, List[Tuple[str, torch.nn.Module]]]]:
        _model_info = ModelInfo(self._model)
        return _model_info.id_to_layerstack_map()

    def _dry_run(self):
        # Copying a model will change the layer name, use input model directly here.
        _traced_model = torch.fx.symbolic_trace(self._model)
        if self._loss is not None and self._label_tensor is not None:
            self._insert_loss_function(_traced_model)
            _traced_model.recompile()
        _interpreter = _AnalysisInterpreter(_traced_model, id2layer=self._build_layer_call_stack())
        if self._loss is not None and self._label_tensor is not None:
            _interpreter.run(self._input_tensor, self._label_tensor)
        else:
            _interpreter.run(self._input_tensor)

        self._layer_info = _interpreter.op_nodes

    def _insert_loss_function(self, model: torch.fx.GraphModule):
        loss_name = self._loss.__class__.__name__
        model.add_module(loss_name, self._loss)
        graph = model.graph
        with graph.inserting_after(next(iter(graph.nodes))):  # 在图的最前面插入占位符
            target_placeholder = graph.create_node('placeholder', target='y', name='y')
        for node in graph.nodes:
            if node.op == "output":
                original_output_node = node.args[0]
                with graph.inserting_before(node):
                    loss_node = graph.create_node(
                        "call_module",
                        target=loss_name,
                        args=(original_output_node, target_placeholder),
                        name="loss"
                    )
                node.args = (loss_node,)
                break

    def layer_summary(self, dnnmem_format: bool = False) -> List[Dict[str, Any]]:
        """ Returns the layer summary of the model

        Examples:
            [
                {
                    'name': 'Conv2d',
                    'stack': 'DenseNet_0->Sequential_0->Conv2d_0',
                    'input': [torch.Size([1, 3, 86, 86])],
                    'output': torch.Size([1, 64, 43, 43]),
                    'weight': torch.Size([64, 3, 7, 7]),
                    'bias': None,
                    'is_inplace': False
                }
                ...
            ]

        Returns:
            List[Dict[str, Any]]: Layer

        """
        layer_dict = []
        for name, value in self.layers_info.items():
            if dnnmem_format:
                info = value.dnnmem_format()
            else:
                info = value.to_json()
            layer_dict.append(info)
        return layer_dict

