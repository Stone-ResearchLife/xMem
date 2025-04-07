import torch
import copy
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, Dict

from sympy.printing.cxx import reserved

from .profiler import ProfilerDataProcessing, MemoryBlock, CpuInstantNode
from .allocator import AllocatorSim, CachingAllocator
from .config import Config


class _Estimator(ABC):
    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        profiler_file: str,
        max_gpu_memory_in_gb: int = 8,
        config: Optional[Config] = None,
    ):
        self.config = config or Config()
        self.dataloader = dataloader
        self.profiler = ProfilerDataProcessing(profiler_file)
        self.allocator_sim = AllocatorSim(
            max_allocated_memory_gb=max_gpu_memory_in_gb, config=self.config
        )
        self._model_memory: Optional[List[MemoryBlock]] = None

    @abstractmethod
    def model_memory(
        self, iteration_index: int, reset: bool = False, *args, **kwargs
    ) -> List[MemoryBlock]:
        pass

    @abstractmethod
    def data_memory(
        self, iteration_index: int, need_released: bool = False, *args, **kwargs
    ) -> List[MemoryBlock]:
        pass

    def training_memory(
        self, iteration_index: int, zero_grad: bool = False, *args, **kwargs
    ) -> List[MemoryBlock]:
        assert isinstance(iteration_index, int)
        assert iteration_index > 0
        memory_activity: List[MemoryBlock] = []
        iteration = self.profiler.get_iteration(iteration_index)
        zero_grad_time = iteration.zero_grad_time[0]
        if zero_grad_time is not None and zero_grad_time > iteration.optimiser_step[1]:
            # zero_grad > end of optimiser step time means that all gradients are zeroed after optimisation
            # optimiser.zero_grad() is called in every end of the iteration
            # Thus, manually adjusting free time of the gradients is unnecessary
            zero_grad = False

        layers = iteration.layer_summary()
        if not zero_grad:  # zero_grad is False
            for mem in layers:
                for forward in mem.get("forward_memory", []):
                    forward._comments.append(mem["name"])
                    forward._forward = True
                    forward._backward = False
                    memory_activity.append(forward)
                for backward in mem.get("backward_memory", []):
                    backward._comments.append(mem["name"])
                    backward._forward = False
                    backward._backward = True
                    memory_activity.append(backward)
        else:
            next_iteration = self.profiler.get_iteration(iteration_index + 1)
            for mem in layers:
                for forward in mem.get("forward_memory", []):
                    forward._comments.append(mem["name"])
                    forward._forward = True
                    forward._backward = False
                    memory_activity.append(forward)
                for backward in mem.get("backward_memory", []):
                    backward._comments.append(mem["name"])
                    backward._forward = False
                    backward._backward = True
                    if backward.free_time is None:
                        end_mem = self.create_memory_block((-1 * backward.bytes))
                        end_mem_cpu_instant = end_mem._start
                        end_mem_cpu_instant._value["ts"] = (
                            next_iteration.zero_grad_time[0]
                        )
                        end_mem_cpu_instant._value["args"][
                            "Addr"
                        ] = backward._start.address
                        backward.set_free_node(end_mem_cpu_instant)
                    memory_activity.append(backward)
        return sorted(memory_activity, key=lambda x: x.alloc_time)

    def optimiser_memory(
        self, iteration_index: int, persist_required: bool = True, *args, **kwargs
    ) -> List[MemoryBlock]:
        iteration_data = self.profiler.get_iteration(iteration_index)
        ops_memory = iteration_data.optimiser_memory()
        model_mem = [
            mem.bytes
            for mem in self.model_memory(iteration_index=iteration_index, reset=False)
        ]
        filter_ops_memory = []
        for mem in ops_memory:
            if mem.bytes in model_mem:
                new_mem = copy.deepcopy(mem)
                if persist_required:
                    new_mem._end = None
                filter_ops_memory.append(new_mem)
        return filter_ops_memory

    def construct_memory_sequence(self, target_iteration: int = 2) -> List[MemoryBlock]:
        if target_iteration > 1:
            # The memory blocks in the first iteration should be replaced by the memory blocks in the second iteration
            # Because the memory blocks in the first iteration are not freed until the end of the second iteration
            target_iteration_memory = self.profiler.get_iteration(
                target_iteration
            ).get_memory_activities()
            for iteration in range(1, target_iteration):
                _memory = self.profiler.get_iteration(iteration)
                _memory._memory = target_iteration_memory

        # load memory blocks of model's parameters
        memory_activity = copy.deepcopy(
            self.model_memory(iteration_index=target_iteration, reset=True)
        )
        for iteration in range(1, target_iteration + 1):
            if iteration == target_iteration:
                dataset_need_released = False
                zero_grad = False
            else:
                dataset_need_released = True
                zero_grad = True

            if iteration == 1:
                optimiser_required = True
            else:
                optimiser_required = False

            memory_activity += self.data_memory(
                iteration_index=iteration, need_released=dataset_need_released
            )
            memory_activity += self.training_memory(
                iteration_index=iteration, zero_grad=zero_grad
            )
            memory_activity += self.optimiser_memory(
                iteration_index=iteration, persist_required=optimiser_required
            )

        sorted_memory = sorted(memory_activity, key=lambda x: x.alloc_time)

        return sorted_memory

    def create_memory_block(
        self, byte: Union[int, float], timestamp: Optional[int] = None
    ) -> MemoryBlock:
        cpu_instant_data = {
            "ph": "i",
            "cat": "cpu_instant_event",
            "s": "t",
            "name": "[memory]",
            "pid": 0,
            "tid": 0,
            "ts": timestamp or 0,
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
        return memory_block

    def estimate_memory_blocks(
        self, blocks: List[MemoryBlock]
    ) -> Tuple[CachingAllocator, Dict]:
        sim_result: CachingAllocator = self.allocator_sim.simulate(blocks)
        ## supplement the memory section with the memory usage of forward and backward
        peak_seg = max(sim_result._trace.max_segment_changes)
        peak_tensor = max(sim_result._trace.max_usage_changes)
        estimated_result = {
            "OOM": sim_result.oom,
            "Max GPU Memory": sim_result.allowed_memory_maximum,
            "memory": {
                "tensor": peak_tensor,
                "segment": peak_seg,
            },
        }
        return sim_result, estimated_result

    def estimate(self, target_iteration: int = 2) -> Tuple[CachingAllocator, Dict]:
        """Generate memory activity in order of time for the model

        Returns:
            List[MemoryBlock]: a list of memory activity in order of time

        """
        sorted_memory = self.construct_memory_sequence(
            target_iteration=target_iteration
        )
        return self.estimate_memory_blocks(sorted_memory)


class Estimator(_Estimator):
    def model_memory(
        self, iteration_index: int, reset: bool = False, *args, **kwargs
    ) -> List[MemoryBlock]:
        if reset or self._model_memory is None:
            layers_memory = self.training_memory(iteration_index, zero_grad=False)
            layers_memory = copy.deepcopy(layers_memory)
            layers_memory.reverse()
            model_blocks = []
            for index, memory in enumerate(layers_memory):
                if memory.is_backward and memory.free_time is None:
                    model_blocks_start_instant = memory._start
                    model_blocks_start_instant._value["ts"] = index
                    model_blocks.append(memory)
            self._model_memory = model_blocks
        return copy.deepcopy(self._model_memory)

    def data_memory(
        self, iteration_index: int, need_released: bool = False, *args, **kwargs
    ) -> List[MemoryBlock]:
        assert iteration_index > 0
        assert isinstance(iteration_index, int)

        data = next(iter(self.dataloader))
        dataset_memory = [
            self.create_memory_block(tensor.nbytes)
            for tensor in data
            if isinstance(tensor, torch.Tensor)
        ]

        memory_activity: List[MemoryBlock] = []
        iteration = self.profiler.get_iteration(iteration_index)
        loading_time = iteration.dataset_load_time
        for index, mem in enumerate(dataset_memory):
            mem._start._value["ts"] = loading_time[index]
            if need_released:
                end_mem = self.create_memory_block((-1 * mem.bytes))
                end_mem_cpu_instant = end_mem._start
                end_mem_cpu_instant._value["ts"] = iteration.end
                end_mem_cpu_instant._value["args"]["Addr"] = mem._start.address
                mem.set_free_node(end_mem_cpu_instant)
        ## calculate amount of memory size of the dataset
        memory_activity.extend(dataset_memory)
        return memory_activity


class TrainerEstimator(_Estimator):
    def model_memory(
        self, iteration_index: int, reset: bool = False, *args, **kwargs
    ) -> List[MemoryBlock]:
        if reset or self._model_memory is None:
            model_blocks = []
            if (
                self.config.trainer.huggingface_enable
                and self.config.trainer.huggingface_model_name is not None
            ):
                from transformers import AutoConfig, AutoModelForCausalLM

                config = AutoConfig.from_pretrained(
                    self.config.trainer.huggingface_model_name
                )  # load config; do NOT load pretrained weights
                model = AutoModelForCausalLM.from_config(
                    config
                )  # randomly initialized model

                parameters_list = list(model.parameters())
                parameters_list.reverse()
                global_index = 0
                for index, tensor in enumerate(parameters_list):
                    global_index += 1
                    para_size = tensor.nelement() * tensor.element_size()
                    parameter_memory_block = self.create_memory_block(
                        byte=para_size, timestamp=global_index
                    )
                    model_blocks.append(parameter_memory_block)

                buffer_list = list(model.buffers())
                buffer_list.reverse()
                for index, buffer in enumerate(buffer_list):
                    global_index += 1
                    buffer_size = buffer.nelement() * buffer.element_size()
                    block = self.create_memory_block(
                        byte=buffer_size,
                        timestamp=global_index
                    )
                    model_blocks.append(block)

            else:
                zero_time = self.profiler.get_iteration(iteration_index).zero_grad_time
                # process the memory blocks of the model
                layers_memory = self.training_memory(iteration_index, zero_grad=False)
                layers_memory = copy.deepcopy(layers_memory)
                layers_memory.reverse()
                for index, memory in enumerate(layers_memory):
                    if (
                        memory.is_backward
                        and zero_time[0] < memory.free_time < zero_time[1]
                    ):
                        _memory = copy.deepcopy(memory)
                        _memory_start_instant = _memory._start
                        _memory_start_instant._value["ts"] = index
                        # Force to set the end of the memory block to None, treating it as persistent memory
                        _memory._end = None
                        model_blocks.append(_memory)
            self._model_memory = model_blocks
        return copy.deepcopy(self._model_memory)

    def data_memory(
        self, iteration_index: int, need_released: bool = False, *args, **kwargs
    ) -> List[MemoryBlock]:
        assert isinstance(iteration_index, int)
        assert iteration_index > 0
        cpu_ops = self.profiler.get_iteration(iteration_index).cpu_ops
        selected_op = []
        for start, op in cpu_ops:
            if op.function_name == "to":
                concrete_inputs = "|".join(
                    [
                        f"{_in['index']}-{_in['concrete_input']}"
                        for _in in op.concrete_inputs
                    ]
                )
                expect_concrete_inputs = [
                    "1-6|2-0|5-False|6-False",
                    "1-4|2-0|5-False|6-False",
                ]
                if concrete_inputs in expect_concrete_inputs:
                    selected_op.append(op)
        selected_op = selected_op[:-1]
        data_memory = []
        for op in selected_op:
            start_time = op.start_time
            bytes = sum([int(arg["bytes"]) for arg in op.input_args])
            mem = self.create_memory_block(bytes, start_time)
            if iteration_index == self.profiler.max_iterations:
                end_time = self.profiler.get_iteration(iteration_index).end
            else:
                # todo: the end time of the memory block should be after the next batch of data is loaded
                end_time = self.profiler.get_iteration(iteration_index + 1).start
            end_bytes = -1 * bytes
            end_mem = self.create_memory_block(end_bytes, end_time)
            end_mem_cpu_instant = end_mem._start
            end_mem_cpu_instant._value["args"]["Addr"] = mem._start.address
            mem.set_free_node(end_mem_cpu_instant)
            data_memory.append(mem)
        return data_memory
