# ⚖️ Experiments

## ✅ Compatibility

- ✅: Runnable
- ❌: Unrunnable
- ⚠️: Unknown

| **Hardware**  | **Compatibility** |
| :-----------: | :---------------: |
|  NVIDIA GPU   |        ✅         |
|   Intel CPU   |        ✅         |
|    AMD CPU    |        ⚠️         |
|    AMD GPU    |        ⚠️         |
| Apple Silicon |        ❌         |

## 1️⃣ Dependencies

1. Ensure all dependencies are installed, as shown in the [README](../README.md).
2. Ensure Docker Client is installed on your machine. If not follow the instructions [here](https://docs.docker.com/engine/install/).

## 2️⃣ Docker Pull a Base Image

Pull the base image from Docker Hub. This image is used to build images for the experiments.
Image Link: [here](https://hub.docker.com/layers/pytorch/pytorch/2.3.1-cuda12.1-cudnn8-devel/images/sha256-a22a1fca37f8361c8a1e859cd6eb6bd9d1fb384f9c0dcb2cfc691a178eb03d17?context=explore)

```shell
docker pull pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
docker pull pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
docker pull pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
```

## 3️⃣ Build Runtime Image for Experiments

> [!WARNING]
> Ensure you set the right PYTHONPATH before running the experiments. The PYTHONPATH should be the root directory of the project.

Since a special build requirement of LLMem, we have to build base image for LLMem by

>[!TIP]
> You could also build environment yourself via this [link](https://github.com/taehokim20/LLMem)
```bash
cd exp/baselines/LLmem
docker build -t llmem .
```
After build, you will get a image, called `llmem`.
```text
REPOSITORY        TAG                           IMAGE ID       CREATED          SIZE
llmem             latest                        1af477f97e32   54 seconds ago   13.7GB
pytorch/pytorch   2.6.0-cuda12.4-cudnn9-devel   7d57e307bd9c   3 months ago     13.2GB
pytorch/pytorch   2.3.1-cuda12.1-cudnn8-devel   b40b101922fd   11 months ago    17.1GB
pytorch/pytorch   2.0.1-cuda11.7-cudnn8-devel   42a0e9b621e2   2 years ago      13.2GB
```

Next, an auto-build script should be run to build rest of images.

> [!WARNING]
> You have to change workdir to root of project.
```shell
python apps.py experiments prepare 
```

## 4️⃣ Check Runtime Image

Check whether both image are built successfully

```shell
docker images
```

Shown as
```text
REPOSITORY              TAG                           IMAGE ID       CREATED         SIZE
llmem-estimator         latest                        100b183a6cc8   2 minutes ago   13.8GB
schedtun-estimator      latest                        006e83a1937f   2 minutes ago   18.7GB
paper-experiments       latest                        8ae7c8a4b68c   2 minutes ago   18.2GB
paper-estimator         latest                        ebe543382d65   4 minutes ago   13.9GB
paper-experiments-llm   latest                        0eadd4567983   4 minutes ago   13.8GB
llmem                   latest                        1af477f97e32   9 minutes ago   13.7GB
pytorch/pytorch         2.6.0-cuda12.4-cudnn9-devel   7d57e307bd9c   3 months ago    13.2GB
pytorch/pytorch         2.3.1-cuda12.1-cudnn8-devel   b40b101922fd   11 months ago   17.1GB
pytorch/pytorch         2.0.1-cuda11.7-cudnn8-devel   42a0e9b621e2   2 years ago     13.2GB
```


## 5️⃣ Run Experiments


>[!IMPORTANT]
> Please ensure that 
> - PyTorch is installed with a CUDA version
> - `notebook` has been installed in environment

> [!WARNING]
> Do not run any GPU-related tasks on the GPUs used during the experiment, as they will be occupied for specific purposes.

> [!WARNING]
> Multiple directories are created by experiments:
> - `~/CNN-Exp` to store result related to CNN models
> - `~/Transformer-Exp` to store result related to Transformer Models
> - `~/Large-Transformer-Exp` to store result related to Qwen3 0.6B and Pythia 1B

### Run ANOVA Experiment

Use a Jupyter [Notebook](Experiments-ANOVA.ipynb) for this experiments 


### Run Monte Carlo Experiment
Use a Jupyter [Notebook](Experiments-Mento%20Carlo.ipynb) for this experiments 



## 6️⃣ Plot Results

Please follow the instructions in the [README](../plot/README.md) to plot the results.

## 7️⃣ Clean Up (Optional)

> [!WARNING]
>
> - The command will remove all stopped containers and all dangling images
> - Please do not execute this command if you have concerns about the code, as it involves a `delete` operation.

```shell
python app.py cleanup
```

## Our Results

You could create all figures in the paper by following the instructions in the [README](../plot/README.md).

- ANOVA Result: `<root project>/plot/data/001-ANOVA`
- Monte Carlo Result: `<root project>/plot/data/002-Monte Carlo`
- Memory Change Result: `<root project>/plot/data/003-MemoryChange`
- Simulation Assessment Result: `<root project>/plot/data/004-SimulationAssessment`
- Scalability Result: `<root project>/plot/data/005-ScalabilityOnCoLab`

# 📚 Resources
- [SchedTune Source Code](https://github.com/hadeelalbahar/SchedTune)
- [LLMem Source Code](https://github.com/taehokim20/LLMem)
