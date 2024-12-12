# ⚖️ Experiments


## ✅ Compatibility
- ✅: checked
- ❌: not work
- ⚠️: Not Checked

| **Hardware** | **Compatibility** |
|:------------:|:-----------------:|
|  NVIDIA GPU  |         ✅         |
|  Intel CPU   |         ✅         |
|   AMD CPU    |        ⚠️         |
|   AMD GPU    |        ⚠️         |
|     Mac      |         ❌         | 



## 1️⃣ Dependencies
1. Ensure all dependencies are installed, as shown in the [README](../README.md).
2. Ensure Docker Client is installed on your machine. If not follow the instructions [here](https://docs.docker.com/engine/install/).


## 2️⃣ Docker Pull a Base Image
Pull the base image from Docker Hub. This image is used to build images for the experiments.
Image Link: [here](https://hub.docker.com/layers/pytorch/pytorch/2.3.1-cuda12.1-cudnn8-devel/images/sha256-a22a1fca37f8361c8a1e859cd6eb6bd9d1fb384f9c0dcb2cfc691a178eb03d17?context=explore)
```shell
docker pull pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
```

## 3️⃣ Build Runtime Image for Experiments

> [!WARNING]
> Ensure you set the right PYTHONPATH before running the experiments. The PYTHONPATH should be the root directory of the project.

Execute a pre-configured Python script to build two images: 
- one is the base image called `base-xmem-evaluation`
- the other is the runtime image called `xmem-evaluation`.

```shell
python evaluate.py xMem build
```
Forcely rebuild the image by adding `--force` at the end of the command.

## 4️⃣ Check Runtime Image
Check whether both image are built successfully
```shell
docker images
```
Shown as
```text
REPOSITORY                              TAG                           IMAGE ID       CREATED          SIZE
xmem-evaluation                         latest                        598032aa0a09   6 minutes ago    18.4GB
base-xmem-evaluation                    latest                        78fd57f2ba85   6 minutes ago    18.3GB
```

## 5️⃣ Run Experiments
> [!WARNING]
> Do not run any GPU-related tasks on the GPUs used during the experiment, as they will be occupied for specific purposes.

> [!WARNING]
> Two directories are created by experiments:
> - `~/.cache/xMemExperiments` to store results
> - `~/pytorch_datasets` to store datasets used in the experiments.

### Run ANOVA Experiment
Run the below command to execute the ANOVA experiment, which may take 1-3 days depending on the performance of the machine.

The parameter `-g` is used to set the GPU index. 
If you have multiple GPUs, you can specify which one to use for running the experiment.Default is 0.

```shell
python evaluate.py xMem anova -g 0
```

### Run Monte Carlo Experiment
> [!NOTE]
> The container may be likely crashed by SchedTune when running the Monte Carlo experiment 
> with a combination of large batch sizes (>900) and large models (RegNetX32GF, RegNetY32GF or ResNet152).
> it leads to a non-zero exit code of container and no results generated.

Execute the Monte Carlo experiment using the command below. 
This process may take days, depending on the total number of repetitions you want to run, 
which can be passed as an argument to the script with `-t`.


In our experiments, there are two GPUs involved, so we set `devices=[0,1]` under `monte_carlo` method in the `evaluate.py`.
Otherwise, you can set `devices=[0]` if you only have one GPU.
```shell
python evaluate.py xMem monte_carlo -t 1000
```

## 6️⃣ Plot Results
Please follow the instructions in the [README](../plot/README.md) to plot the results.

### Location of the Results
By default, all results are stored in the `~/.cache/xMemExperiments` directory.
The directory is created during the run automatically.


## 7️⃣ Clean Up (Optional)
> [!WARNING]
> - The command will remove all stopped containers and all dangling images 
> - Please do not execute this command if you have concerns about the code, as it involves a `delete` operation.
```shell
python evaluate.py cleanup
```

# 🧩 Environment we used
|    **Components**     |      **Version/SKU**      | **Comments** |
|:---------------------:|:-------------------------:|:-----------:|
|          CPU          |   Intel® Core™ i9-13900   |   24 cores  | 
|     GPU (index:0)     | NVIDIA GeForce RTX 4070Ti | 12GB GDDR6X |
|     GPU (index:1)     | NIVIDIA GeForce RTX 4060  |  8GB GDDR6  |
|          RAM          |        128GB DDR4         |  5400MHz    |
|          OS           |   Ubuntu 22.04.3 x86_64   |             |
|     Docker (Host)     |          27.3.1           |             |
|  CUDA Version (Host)  |           12.2            |             |
| cuDNN Version (Host)  |           8.9.7           |             |
| NVIDIA Driver Version |         555.42.06         |             |

## Our Results
You could create all figures in the paper by following the instructions in the [README](../plot/README.md).
- ANOVA Result: `<root project>/plot/data/ANOVA`
- Monte Carlo Result: `<root project>/plot/data/MonteCarlo`
- Memory Change Result: `<root project>/plot/data/MemoryChange`
- Simulation Assessment Result: `<root project>/plot/data/SimulationAssessment`


# 📚 Resources
- [SchedTune Source Code](https://github.com/hadeelalbahar/SchedTune)
- [LLMem Source Code](https://github.com/taehokim20/LLMem)
