import time
import fire
import uuid
import tqdm
from pathlib import Path
from typing import Union, Optional
from experiments.scripts import DockerCleanup, InterfaceApp, Evaluation
from perf_estimator.utilis.enum import EnumManipulator


class XMemEvaluation(InterfaceApp):
    def __init__(self, temp_dir: Optional[Union[str, Path]] = None):
        self._temp_dir = temp_dir
        if self._temp_dir is None:
            _out_dir = Path().home().joinpath(".cache/xMemExperiments")
            self._temp_dir = _out_dir
        self._temp_dir = Path(self._temp_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._image_name = "xmem-evaluation"
        self._dataset_dir = Path().home().joinpath("pytorch_datasets")

    def build(self, force=False):
        estimator = Evaluation(
            output_dir=self._temp_dir,
            container_name=self._image_name,
        )
        estimator.build_image(force=force)

    def _execute(
            self,
            model: str = "VGG16",
            device: int = 0,
            batch_size: int = 200,
            iteration: int = 2,
            optimiser: str = "SGD",
            zero_grad_mode: int = 0,
            verification: bool = False,
            input_size: int = 86,
    ):
        estimator = Evaluation(
            output_dir=self._temp_dir,
            container_name=self._image_name,
            dataset_dir=self._dataset_dir
        )
        command = [
            "-m", model,
            "-d", device,
            "-b", batch_size,
            "-t", iteration,
            "-o", optimiser,
            "-z", zero_grad_mode,
            "-v", verification,
            "-i", input_size,
        ]
        name = f"xmen-{uuid.uuid4().hex[:8]}"
        container = estimator.run_image(image_name=self._image_name, name=name, command=command)
        return container

    def execute(
            self,
            model: str = "VGG16",
            device: int = 0,
            batch_size: int = 200,
            target_iteration: int = 2,
            optimiser: str = "SGD",
            zero_grad_mode: int = 1
    ):
        _contain = self._execute(
            model=model,
            device=device,
            batch_size=batch_size,
            iteration=target_iteration,
            optimiser=optimiser,
            zero_grad_mode=zero_grad_mode
        )


    def anova(self, gpu: int = 0):
        from perf_estimator.models import AllModels
        enum_op = EnumManipulator(AllModels)
        models = enum_op.fetch_keys()
        device = gpu
        times = 3
        original_tmp = self._temp_dir
        optimiser = ["SGD", "Adam", "RMSprop","Adagrad", "AdamW"]
        for opt in tqdm.tqdm(optimiser):
            for batch_size in tqdm.tqdm([10, 50, 90, 130, 170, 210, 250, 290, 330, 370, 410, 450, 490, 530]):
                for model in tqdm.tqdm(models):
                    self._temp_dir = original_tmp.joinpath(f"anova-{model}-{opt}-{batch_size}-{device}")
                    self._temp_dir.mkdir(parents=True, exist_ok=True)
                    for i in tqdm.tqdm(range(times)):
                        _contain = self._execute(
                            model=model,
                            device=device,
                            batch_size=batch_size,
                            iteration=2,
                            optimiser=opt,
                            zero_grad_mode=0,
                            verification=True
                        )

                        time.sleep(2)

    def monte_carlo(self, times: int = 10):
        import random
        from perf_estimator.models import AllModels
        enum_op = EnumManipulator(AllModels)
        models = enum_op.fetch_keys()
        devices = [0, 1]
        original_tmp = self._temp_dir
        optimiser = ["SGD", "Adam", "RMSprop", "Adagrad", "AdamW"]
        input_sizes = [86]
        zero_grad_modes = [0, 1]

        for i in tqdm.tqdm(range(times)):
            model = str(random.choice(models))
            device = int(random.choice(devices))
            batch_size = random.randint(10, 1000)
            opt = str(random.choice(optimiser))
            zero_grad_mode = int(random.choice(zero_grad_modes))
            input_size = int(random.choice(input_sizes))

            self._temp_dir = original_tmp.joinpath(f"MonteCarlo-{model}-{opt}-{batch_size}-{device}")
            self._temp_dir.mkdir(parents=True, exist_ok=True)
            _contain = self._execute(
                model=model,
                device=device,
                batch_size=batch_size,
                iteration=2,
                optimiser=opt,
                zero_grad_mode=zero_grad_mode,
                verification=True,
                input_size=input_size
            )
            time.sleep(2)


class AppSet:
    def __init__(self):
        self.xMem = XMemEvaluation()

    def cleanup(self):
        _cleanup = DockerCleanup()
        _cleanup.stopped_containers()
        _cleanup.dangling_images()


if __name__ == '__main__':
    fire.Fire(AppSet)