import torch
from typing import Optional
from .trainer import ModelPreparer
from .config import CNNExperments, TransformerExperiments


class ExperimentRun:
    def run_estimator(
            self,
            estimator_name,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            max_est_memory_in_bytes: int,
            optimizer: Optional[type(torch.optim)] = None,
            is_transformer: bool = False
    ):
        estimator_name = estimator_name.lower()
        if estimator_name == "dnnmem":
            from .baselines.dnnmem import DNNmem
            est = DNNmem(
                model=model,
                dataloader=dataloader,
                max_est_memory_in_bytes=max_est_memory_in_bytes,
                optimizer=optimizer,
                is_transformer=is_transformer
            )
        est.estimate()
        return est.estimate_memory, est.execute_time


    def run_cnn_experiment(self):
        """
        Run CNN experiments with the given configuration.
        """
        project_name = "CNN-Experiments"
        conf = CNNExperments()
        for model in conf.models:
            for opt in conf.optimisers:
                for batch_number in range(conf.batch_range[0], conf.batch_range[1], conf.batch_range[2]):
                    model_p = ModelPreparer(
                        model_name=model,
                        batch_size=batch_number,
                        optimiser=opt
                    )
                    # Estimate by DNNmem
                    print(f"Starting estimation for {model}-{opt}-{batch_number}")
                    estimator_name = "dnnmem"
                    mem, runtime = self.run_estimator(
                        estimator_name=estimator_name,
                        model=model_p.model,
                        dataloader=model_p.dl,
                        max_est_memory_in_bytes=8*1024**3,
                        optimizer=model_p.optimiser,
                        is_transformer=False
                    )
                    print(f"Model: {model}-{opt}{batch_number}, Memory: {mem}, Runtime: {runtime}")





