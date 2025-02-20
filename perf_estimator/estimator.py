import torch
import copy
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, Dict
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
        self.dataloader = dataloader
        self.profiler = ProfilerDataProcessing(profiler_file)
        self.allocator_sim = AllocatorSim(
            max_allocated_memory_gb=max_gpu_memory_in_gb, config=config
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

    @abstractmethod
    def optimiser_memory(
        self, iteration_index: int, persist_required: bool = True, *args, **kwargs
    ) -> List[MemoryBlock]:
        pass

    def training_memory(
        self, iteration_index: int, zero_grad: bool = False, *args, **kwargs
    ) -> List[MemoryBlock]:
        assert isinstance(iteration_index, int)
        assert iteration_index > 0
        memory_activity: List[MemoryBlock] = []
        iteration = self.profiler.get_iteration(iteration_index)
        zero_grad_time = iteration.zero_grad_time
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
                        end_mem_cpu_instant._value["ts"] = next_iteration.zero_grad_time
                        end_mem_cpu_instant._value["args"][
                            "Addr"
                        ] = backward._start.address
                        backward.set_free_node(end_mem_cpu_instant)
                    memory_activity.append(backward)
        return sorted(memory_activity, key=lambda x: x.alloc_time)

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
            current_iteration_zero_grad_time = self.profiler.get_iteration(
                iteration_index
            ).zero_grad_time
            model_blocks = []
            for index, memory in enumerate(layers_memory):
                if memory.is_backward and (
                    memory.free_time is None
                    or memory.free_time == current_iteration_zero_grad_time
                ):
                    model_blocks_start_instant = memory._start
                    model_blocks_start_instant._value["ts"] = index
                    model_blocks.append(memory)
            self._model_memory = model_blocks
        return copy.deepcopy(self._model_memory)

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


# class Estimator:
#     def __init__(
#         self,
#         dataloader: torch.utils.data.DataLoader,
#         profiler_file: str,
#         max_gpu_memory_in_gb: int = 8,
#         config: Optional[Config] = None,
#     ):
#         self.dataloader = dataloader
#         self.profiler = ProfilerDataProcessing(profiler_file)
#         self.allocator_sim = AllocatorSim(
#             max_allocated_memory_gb=max_gpu_memory_in_gb, config=config
#         )
#         self._model_memory: Optional[List[MemoryBlock]] = None
#
#     def model_memory(self, iteration: int, reset: bool = False) -> List[MemoryBlock]:
#         if reset or self._model_memory is None:
#             layers_memory = self.get_an_iteration_memory(iteration, zero_grad=False)
#             layers_memory = copy.deepcopy(layers_memory)
#             layers_memory.reverse()
#             current_iteration_zero_grad_time = self.profiler.get_iteration(
#                 iteration
#             ).zero_grad_time
#             model_blocks = []
#             for index, memory in enumerate(layers_memory):
#                 if memory.is_backward and (
#                     memory.free_time is None
#                     or memory.free_time == current_iteration_zero_grad_time
#                 ):
#                     model_blocks_start_instant = memory._start
#                     model_blocks_start_instant._value["ts"] = index
#                     model_blocks.append(memory)
#             self._model_memory = model_blocks
#         return copy.deepcopy(self._model_memory)
#
#     def data_memory(self) -> List[MemoryBlock]:
#         data = next(iter(self.dataloader))
#         return [
#             self.create_memory_block(tensor.nbytes)
#             for tensor in data
#             if isinstance(tensor, torch.Tensor)
#         ]
#
#     def create_memory_block(self, byte: Union[int, float]) -> MemoryBlock:
#         cpu_instant_data = {
#             "ph": "i",
#             "cat": "cpu_instant_event",
#             "s": "t",
#             "name": "[memory]",
#             "pid": 0,
#             "tid": 0,
#             "ts": 0,
#             "args": {
#                 "Bytes": byte,
#                 "Addr": 0x0,
#                 "Device Id": -1,
#                 "Device Type": 0,
#                 "Ev Idx": 0,
#             },
#         }
#         cpu_instant_node = CpuInstantNode(cpu_instant_data)
#         memory_block = MemoryBlock(cpu_instant_node)
#         return memory_block
#
#     def optimiser_memory(
#         self, iterations=1, persist_required: bool = True
#     ) -> List[MemoryBlock]:
#         mem = self.loss_memory(iterations, persist_required)
#         return mem
#
#     def loss_memory(
#         self, iteration=1, persist_required: bool = True
#     ) -> List[MemoryBlock]:
#         iteration_data = self.profiler.get_iteration(iteration)
#         ops_memory = iteration_data.optimiser_memory()
#         model_mem = [
#             mem.bytes for mem in self.model_memory(iteration=iteration, reset=False)
#         ]
#         filter_ops_memory = []
#         for mem in ops_memory:
#             if mem.bytes in model_mem:
#                 new_mem = copy.deepcopy(mem)
#                 if persist_required:
#                     new_mem._end = None
#                 filter_ops_memory.append(new_mem)
#         return filter_ops_memory
#
#     def get_dataset_memory(
#         self, iteration_index: int, need_released: bool = False
#     ) -> List[MemoryBlock]:
#         assert iteration_index > 0
#         assert isinstance(iteration_index, int)
#         memory_activity: List[MemoryBlock] = []
#         iteration = self.profiler.get_iteration(iteration_index)
#         loading_time = iteration.dataset_load_time
#         dataset_memory = self.data_memory()
#         for index, mem in enumerate(dataset_memory):
#             mem._start._value["ts"] = loading_time[index]
#             if need_released:
#                 end_mem = self.create_memory_block((-1 * mem.bytes))
#                 end_mem_cpu_instant = end_mem._start
#                 end_mem_cpu_instant._value["ts"] = iteration.end
#                 end_mem_cpu_instant._value["args"]["Addr"] = mem._start.address
#                 mem.set_free_node(end_mem_cpu_instant)
#         ## calculate amount of memory size of the dataset
#         memory_activity.extend(dataset_memory)
#         return memory_activity
#
#     def get_an_iteration_memory(
#         self, iteration_index: int, zero_grad: bool = False
#     ) -> List[MemoryBlock]:
#         assert iteration_index > 0
#         assert isinstance(iteration_index, int)
#         memory_activity: List[MemoryBlock] = []
#         iteration = self.profiler.get_iteration(iteration_index)
#         layers = iteration.layer_summary()
#         if not zero_grad:  # zero_grad is False
#             for mem in layers:
#                 for forward in mem.get("forward_memory", []):
#                     forward._comments.append(mem["name"])
#                     forward._forward = True
#                     forward._backward = False
#                     memory_activity.append(forward)
#                 for backward in mem.get("backward_memory", []):
#                     backward._comments.append(mem["name"])
#                     backward._forward = False
#                     backward._backward = True
#                     memory_activity.append(backward)
#         else:
#             next_iteration = self.profiler.get_iteration(iteration_index + 1)
#             for mem in layers:
#                 for forward in mem.get("forward_memory", []):
#                     forward._comments.append(mem["name"])
#                     forward._forward = True
#                     forward._backward = False
#                     memory_activity.append(forward)
#                 for backward in mem.get("backward_memory", []):
#                     backward._comments.append(mem["name"])
#                     backward._forward = False
#                     backward._backward = True
#                     if backward.free_time is None:
#                         end_mem = self.create_memory_block((-1 * backward.bytes))
#                         end_mem_cpu_instant = end_mem._start
#                         end_mem_cpu_instant._value["ts"] = next_iteration.zero_grad_time
#                         end_mem_cpu_instant._value["args"][
#                             "Addr"
#                         ] = backward._start.address
#                         backward.set_free_node(end_mem_cpu_instant)
#                     memory_activity.append(backward)
#         return sorted(memory_activity, key=lambda x: x.alloc_time)
#
#     def get_memory_activity(self, target_iteration: int = 2) -> List[MemoryBlock]:
#         if target_iteration > 1:
#             # The memory blocks in the first iteration should be replaced by the memory blocks in the second iteration
#             # Because the memory blocks in the first iteration are not freed until the end of the second iteration
#             target_iteration_memory = self.profiler.get_iteration(
#                 target_iteration
#             ).get_memory_activities()
#             for iteration in range(1, target_iteration):
#                 _memory = self.profiler.get_iteration(iteration)
#                 _memory._memory = target_iteration_memory
#
#         # load memory blocks of model's parameters
#         memory_activity = self.model_memory(target_iteration, reset=True)
#         for iteration in range(1, target_iteration + 1):
#             if iteration == target_iteration:
#                 dataset_need_released = False
#                 zero_grad = False
#             else:
#                 dataset_need_released = True
#                 zero_grad = True
#
#             if iteration == 1:
#                 optimiser_required = True
#             else:
#                 optimiser_required = False
#
#             memory_activity += self.get_dataset_memory(
#                 iteration_index=iteration, need_released=dataset_need_released
#             )
#             memory_activity += self.get_an_iteration_memory(
#                 iteration_index=iteration, zero_grad=zero_grad
#             )
#             memory_activity += self.optimiser_memory(
#                 iterations=iteration, persist_required=optimiser_required
#             )
#
#         sorted_memory = copy.deepcopy(
#             sorted(memory_activity, key=lambda x: x.alloc_time)
#         )
#         return sorted_memory
#
#     def estimate(self, target_iteration: int = 2) -> Tuple[CachingAllocator, Dict]:
#         """Generate memory activity in order of time for the model
#
#         Returns:
#             List[MemoryBlock]: a list of memory activity in order of time
#
#         """
#         sorted_memory = self.get_memory_activity(target_iteration=target_iteration)
#         return self.memory_estimate(sorted_memory)
#
#     def _calculate_input_tensor_size(self) -> int:
#         _data = self.data_memory()
#         # only pick X tensor for image classification training.
#         # Actually, tensor Y does not impact the memory usage too much.
#         _input_data = _data[0]
#         return _input_data.bytes
#
#     def memory_estimate(
#         self, memory_activities: List[MemoryBlock]
#     ) -> Tuple[CachingAllocator, Dict]:
#         sim_result: CachingAllocator = self.allocator_sim.simulate(memory_activities)
#         ## supplement the memory section with the memory usage of forward and backward
#         peak_seg = max(sim_result._trace.max_segment_changes)
#         peak_tensor = max(sim_result._trace.max_usage_changes)
#         estimated_result = {
#             "OOM": sim_result.oom,
#             "Max GPU Memory": sim_result.allowed_memory_maximum,
#             "memory": {
#                 "tensor": peak_tensor,
#                 "segment": peak_seg,
#             },
#         }
#         max_gpu_memory_in_gp = sim_result.allowed_memory_maximum / 1024**3
#         try:
#             input_size = self._calculate_input_tensor_size()
#             seg_memory_overhead_pre_byte = (
#                 peak_seg / input_size
#             ) / max_gpu_memory_in_gp
#             tensor_memory_overhead_pre_byte = (
#                 peak_tensor / input_size
#             ) / max_gpu_memory_in_gp
#         except Exception as e:
#             seg_memory_overhead_pre_byte = -1
#             tensor_memory_overhead_pre_byte = -1
#             print(f"Error: {e}")
#         finally:
#             estimated_result["memory"]["seg_memory_overhead_pre_byte"] = round(
#                 seg_memory_overhead_pre_byte, 0
#             )
#             estimated_result["memory"]["tensor_memory_overhead_pre_byte"] = round(
#                 tensor_memory_overhead_pre_byte, 0
#             )
#
#         return sim_result, estimated_result
