import torch
import copy
import time
from typing import Optional
from pathlib import Path
from .estimator import Estimate as SchedtuneEstimator
from ..abc import EstimatorInterface
from ..utils import GenComputationalGraph
from ..fx import FXAnalyser


class ScheduleTune(EstimatorInterface):
    def __init__(
            self,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            optimizer: Optional[type(torch.optim)] = None,
            is_transformer: bool = False,
            device_id: int = 0,
    ):
        activation_size = 0
        parameter_size = 0
        for param in model.parameters():
            parameter_size += param.nelement() * param.element_size()

        if is_transformer:
            input_size = sum([t.nbytes for t in next(iter(dataloader)).values()])
            cg = GenComputationalGraph(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
            )
            cg.prepare_computational_graph_data()
            for tensor in cg.tensor_dict.values():
                if tensor.is_activation_tensor:
                    activation_size += tensor.bytes
        else:
            data_x, data_y = next(iter(copy.deepcopy(dataloader)))
            model_analysis = FXAnalyser(
                model=model, data_x=data_x, data_y=data_y
            )
            input_size = data_x.nbytes + data_y.nbytes
            for key, layer in model_analysis.layers_info.items():
                if layer.op_type == "call_module":
                    if isinstance(layer.output, torch.Tensor):
                        activation_size += layer.output.nbytes

        self.parameter_size = parameter_size
        self.activation_size = activation_size
        self.input_size = input_size
        self.gpu_name = "5080" if device_id == 0 else "4060"
        self.conf_dir = Path(__file__).parent.joinpath("src")
        self._memory = None
        self._execute_time = None

    @property
    def execute_time(self):
        return self._execute_time

    @property
    def estimate_memory(self):
        return float(self._memory)

    def estimate(self, *args, **kwargs) -> None:
        s_time = time.time_ns()
        schedtune = SchedtuneEstimator(
            jobname="batchsize",
            option="1",
            activations=self.activation_size / 1024 ** 2,
            parameters=self.parameter_size / 1024 ** 2,
            inputsize=self.input_size / 1024 ** 2,
            gpu=self.gpu_name,
            conf_dir=self.conf_dir
        )
        schedtune_output = schedtune.estimate()
        e_time = time.time_ns()
        self._execute_time = e_time - s_time
        self._memory =  schedtune_output["mem"] * 1024 ** 2

