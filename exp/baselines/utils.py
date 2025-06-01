import torch
import time
import copy
from typing import Optional, Union, Dict
from perf_estimator.profiler.memoy import MemoryBlock, CpuInstantNode


class _SimulatorTensor:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self._called = []
        # The value of attributes below are only allowed to assign when it is created
        # In order to simplefy the code, we do not set any restriction on them
        self.is_forward = None
        self.is_backward = None
        self.is_output = None

    @property
    def id(self):
        return id(self.tensor)

    @property
    def start(self):
        return min(self._called) if len(self._called) > 0 else None

    @property
    def end(self):
        end_time = max(self._called) if len(self._called) > 0 else None
        if end_time is None:
            return None
        else:
            if end_time == self.start:
                return None
            else:
                return end_time

    @property
    def is_activation_tensor(self) -> bool:
        return all(
            [
                self.is_forward,
                self.is_output
            ]
        )

    @property
    def bytes(self) -> int:
        return self.tensor.nelement() * self.tensor.element_size()

    def record_time(self):
        self._called.append(time.time_ns())


class GenComputationalGraph:
    def __init__(
            self,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            optimizer: Optional[type(torch.optim.Optimizer)] = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.op = optimizer or torch.optim.SGD
        self.tensor_dict: Dict[str, _SimulatorTensor] = {}

    def create_memory_block(
            self,
            byte: Union[int, float],
            start: Optional[int],
            end: Optional[int] = None,
    ) -> MemoryBlock:
        cpu_instant_data = {
            "ph": "i",
            "cat": "cpu_instant_event",
            "s": "t",
            "name": "[memory]",
            "pid": 0,
            "tid": 0,
            "ts": start,
            "args": {
                "Bytes": byte,
                "Addr": 0x0,
                "Device Id": -1,
                "Device Type": 0,
                "Ev Idx": 0,
            },
        }
        cpu_instant_node = CpuInstantNode(cpu_instant_data)
        memory_block = MemoryBlock(cpu_instant_node)

        if end is not None:
            cpu_instant_data_end = {
                "ph": "i",
                "cat": "cpu_instant_event",
                "s": "t",
                "name": "[memory]",
                "pid": 0,
                "tid": 0,
                "ts": end,
                "args": {
                    "Bytes": -1 * byte,
                    "Addr": 0x0,
                    "Device Id": -1,
                    "Device Type": 0,
                    "Ev Idx": 0,
                },
            }
            cpu_instant_node_end = CpuInstantNode(cpu_instant_data_end)
            memory_block.set_free_node(cpu_instant_node_end)
        return memory_block

    def gen_memory_blocks(self):
        memory_blocks = []
        for t_id, tensor in self.tensor_dict.items():
            b_mem = self.create_memory_block(
                byte=tensor.bytes,
                start=tensor.start,
                end=tensor.end
            )
            memory_blocks.append(b_mem)
        return memory_blocks

    def _add_tensor_to_dict(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            _tensor = _SimulatorTensor(tensor)
            _tensor.is_forward = True
            _tensor.is_backward = False
            _tensor.is_output = False
            if _tensor.id not in self.tensor_dict.keys():
                self.tensor_dict[_tensor.id] = _tensor
            else:
                _tensor = self.tensor_dict[_tensor.id]
            _tensor.record_time()

    def _gen_model_memory_block(self):
        model = self.model
        parameters_list = list(model.parameters())
        for index, tensor in enumerate(parameters_list):
            self._add_tensor_to_dict(tensor)

        buffer_list = list(model.buffers())
        for index, buffer in enumerate(buffer_list):
            self._add_tensor_to_dict(buffer)

    def _gen_input_memory_block(self):
        ld = copy.deepcopy(self.dataloader)
        ld1 = next(iter(ld))
        for key, tensor in ld1.items():
            self._add_tensor_to_dict(tensor)

    def prepare_computational_graph_data(self):
        self._gen_model_memory_block()
        self._gen_input_memory_block()
        def forward_hook(module, inputs, output):
            # Optionally: print or log details
            for ten in inputs:
                self._add_tensor_to_dict(ten)
            self._add_tensor_to_dict(output)

        def backward_hook(module, grad_inputs, grad_outputs):
            # Save gradients coming into (grad_inputs) and going out (grad_outputs) of the module.
            for ten in grad_inputs:
                self._add_tensor_to_dict(ten)

            for ten in grad_outputs:
                self._add_tensor_to_dict(ten)

        model = self.model
        dataloader = self.dataloader
        if self.op.__name__ == "Adafactor":
            optimizer = self.op(model.parameters(), lr=1e-5, relative_step=False)
        else:
            optimizer = self.op(model.parameters(), lr=1e-5)
        for name, module in model.named_modules():
            # Only attach hooks to leaf modules (modules without children)
            if len(list(module.children())) == 0:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)

        device = torch.device("cpu")
        for epoch in range(1):
            for index, batch in enumerate(dataloader):
                with torch.set_grad_enabled(True):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if index == 0:
                    break



