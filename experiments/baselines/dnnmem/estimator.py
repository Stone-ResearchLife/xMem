import torch
from typing import Optional, Dict
import plotly.graph_objects as go
from perf_estimator.data_structure.memory import MemoryBlock
from perf_estimator.utilis.utilis import format_memory
from perf_estimator.allocator import AllocatorSim, CachingAllocator
from experiments.fx import FXAnalyser


class TensorMemoryBlock(MemoryBlock):
    def __init__(self, tensor: torch.Tensor, name: str = None):
        assert isinstance(tensor, torch.Tensor)
        self._tensor = tensor
        self._call_index = []
        self._unreleased = False
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def bytes(self) -> int:
        return self._tensor.nbytes

    @property
    def address(self) -> str:
        return hex(id(self._tensor))

    @property
    def alloc_time(self) -> Optional[int]:
        if len(self._call_index) == 0:
            return None
        return min(self._call_index)

    @property
    def free_time(self) -> Optional[int]:
        if self._unreleased:
            return None
        if self.alloc_time == -1:
            return None
        if len(self._call_index) <= 1:
            return None
        return max(self._call_index)

    @property
    def duration(self) -> Optional[int]:
        if self.alloc_time is not None and self.free_time is not None:
            return self.free_time - self.alloc_time
        return None

    def set_unreleased(self):
        self._unreleased = True

    def add_call_index(self, index: int):
        assert isinstance(index, int)
        self._call_index.append(index)

    def __repr__(self):
        return f"{self.name}: {self.bytes} bytes, Start: {self.alloc_time}, End: {self.free_time}"


class DNNmem:
    def __init__(
            self,
            model: torch.nn.Module,
            data_x: torch.Tensor,
            data_y: torch.Tensor,
            loss_fn: Optional[torch.nn.Module] = None,
            max_gpu_memory: Optional[int] = None
    ):
        self._model = model
        self._data_x = data_x
        self._data_y = data_y
        if loss_fn is None:
            loss_fn = torch.nn.CrossEntropyLoss()
        if max_gpu_memory is None:
            max_gpu_memory = 24
        self._analyser = FXAnalyser(model, data_x, data_y, loss=loss_fn)
        self._layers = self._analyser.layers_info
        # This index map is used for recording relationship
        # between operator and its order in the model.
        self._operator_index = {}
        self._backward_index_map = {}
        self._all_tensors: Dict[str, TensorMemoryBlock] = {}
        self._max_gpu_memory = max_gpu_memory

    @property
    def tensor_memory_blocks(self) -> Dict[str, TensorMemoryBlock]:
        return self._all_tensors

    def _get_forward_tensors(self):
        all_tensors: Dict[str, TensorMemoryBlock] = self._all_tensors
        max_number_layer = len(self._layers)
        for index, (key, value) in enumerate(self._layers.items()):
            # Add all tensors belonging to forward propagation
            self._operator_index[index] = value
            self._backward_index_map[value.real_layer_name] = (max_number_layer * 2 - 1) - index
            _op_type = value.op_type
            if _op_type == "output":
                continue
            elif _op_type not in ["placeholder"]:
                weight_block = self._add_into_tensor_set(value.weight, index=index, name=f"{value.real_layer_name}_weight")
                if weight_block is not None:
                    weight_block.set_unreleased()
                bias_block = self._add_into_tensor_set(value.bias, index=index, name=f"{value.real_layer_name}_bias")
                if bias_block is not None:
                    bias_block.set_unreleased()
                self._add_into_tensor_set(value.output, index=index, name=f"{value.real_layer_name}")
            else:
                placeholder_tensor = self._add_into_tensor_set(value.output, index=index, name=f"{value.real_layer_name}_placeholder")
                if placeholder_tensor is not None:
                    placeholder_tensor.set_unreleased()


            for _input in value.inputs:
                _input_tensor_mem = TensorMemoryBlock(_input.output)
                if _input_tensor_mem.address in all_tensors.keys():
                    all_tensors[_input_tensor_mem.address].add_call_index(index)

    def _get_backward_tensors(self):
        all_tensors: Dict[str, TensorMemoryBlock] = self._all_tensors
        max_number_layer = len(self._layers)
        for index, (key, value) in enumerate(reversed(self._layers.items())):
            index = index + max_number_layer
            self._operator_index[index] = value
            saved_tensors_for_backward = value.get_saved_tensors()
            for _tensor in saved_tensors_for_backward:
                _tensor_mem = TensorMemoryBlock(_tensor)
                if _tensor_mem.address in all_tensors.keys():
                    all_tensors[_tensor_mem.address].add_call_index(index)

            # Add output of backward propagation.
            # Because the output of backward propagation is the gradient of input.
            # So we need to add the input tensor of forward propagation.
            for _input in value.inputs:
                input_shape = _input.output.shape
                new_output_tensor = torch.randn(input_shape)
                tensor_mem = self._add_into_tensor_set(new_output_tensor, index=index, name=f"{value.real_layer_name}_grad")
                tensor_consumer_id = self._backward_index_map[_input.real_layer_name]
                tensor_mem.add_call_index(tensor_consumer_id)

            gradients = value.gradient()
            if gradients is not None:
                for params in gradients.get("parameters", []):
                    _tensor = params[-1].variable
                    new_params_tensor = torch.randn(_tensor.shape)
                    tensor_mem = self._add_into_tensor_set(new_params_tensor, index=index, name=f"{value.real_layer_name}_grad_params")
                    # Add 'optimiser' index to the tensor
                    # Why four times? Because the opertimiser is called after the backward propagation.
                    # Just set a big number to make sure it is larger than the backward propagation.
                    tensor_mem.add_call_index(max_number_layer * 3)

    def _add_into_tensor_set(
            self,
            tensor: torch.Tensor,
            index: int,
            name: Optional[str] = None
    ):
        if tensor is None:
            return
        _tensor_mem = TensorMemoryBlock(tensor, name=name)
        _tensor_mem.add_call_index(index)
        if _tensor_mem.address not in self._all_tensors.keys():
            self._all_tensors[_tensor_mem.address] = _tensor_mem
        else:
            pass
            # print(f"Warning: Duplicate tensor address in {self._operator_index[index].real_layer_name}")
        return _tensor_mem

    def plot(self):
        if len(self._all_tensors) == 0:
            print("Execute the forward and backward analysis first.")
            self._get_forward_tensors()
            self._get_backward_tensors()

        # pre-process the tensor data into uniform format for plotting
        time_slots = [mem.free_time for mem in self.tensor_memory_blocks.values() if mem.duration is not None]
        max_time = max(time_slots)
        plot_block_data = []
        for index, memory in enumerate(self.tensor_memory_blocks.values()):
            start = memory.alloc_time
            dur = max_time - start if memory.duration is None else memory.duration
            bytes = memory.bytes
            plot_block_data.append({
                "x0": start,
                "x1": start + dur,
                "y0": index,
                "y1": index + 0.7,
                "name": memory.name,
                "bytes": bytes
            })

        # plot the memory block data
        fig = go.Figure()
        max_y = len(plot_block_data)
        max_x = max_time
        for plot_block in plot_block_data:
            fig.add_shape(
                type="rect",
                x0=plot_block["x0"],
                y0=plot_block["y0"],
                x1=plot_block["x1"],
                y1=plot_block["y1"],
                line=dict(color="black", width=2),
                fillcolor="blue",
                name=plot_block["name"]
            )

            fig.add_annotation(
                x=(plot_block['x0'] + plot_block['x1']) / 2,
                y=(plot_block['y0'] + plot_block['y1']) / 2,
                text=f"{plot_block['name']}({format_memory(plot_block['bytes'])})",
                showarrow=False,
                font=dict(size=12, color="white"),
            )

            if plot_block['y1'] > max_y:
                max_y = plot_block['y1']
            if plot_block['x1'] > max_x:
                max_x = plot_block['x1']


        fig.update_layout(
            xaxis=dict(range=[0, max_x], title="Time(counts)"),
            yaxis=dict(range=[0, max_y], showticklabels=False),
            showlegend=True,
            height=max_y * 30,
            width=800,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        fig.show()

    def estimate(self) -> CachingAllocator:
        if len(self._all_tensors) == 0:
            print("Execute the forward and backward analysis first.")
            self._get_forward_tensors()
            self._get_backward_tensors()
        _sim = AllocatorSim(self._max_gpu_memory)
        without_relu = []
        for memory_block in self.tensor_memory_blocks.values():
            if "relu" not in memory_block.name.lower():
                without_relu.append(memory_block)
        _result = _sim.simulate(without_relu)
        return _result






