# 🧮 XMem: A Cross-Architecture GPU Memory Estimator

> [!Important]
> The current version is still in prototype. Due to the unoptimized code, it may take a longer execution time.

The widespread adoption of Deep Learning (DL) in diverse application areas has significantly increased the demand for
GPUs. Consequently, GPU resources are scarce and are managed in clusters to maximize resource utilization. However,
this shift introduces new debugging challenges when training DL models on shared clusters particularly Out-Of-Memory (OOM)
errors, an issue commonly reported in industry and academic literature. Existing solutions for avoiding OOM primarily
rely on static analysis of the DL model’s computational graph, or leverage GPU resources directly or indirectly to
estimate the peak memory required for training the given task on the target GPU. Unfortunately, relying on GPUs for
these predictions exacerbates resource contention and increases scheduling challenges. Furthermore, the dynamic nature
of model development limits the accuracy of static analysis to estimate peak memory usage. To address these limitations,
we propose xMem, a novel tool that uses CPU-based analysis to accurately predict the memory required for model training
on a GPU. By eliminating the reliance on GPUs for memory estimation, xMem promotes efficient GPU utilization while
mitigating OOM errors. Our empirical evaluation of 16 DL models (a total of 5,040 runs) demonstrates that, compared to
state-of-the-art GPU memory estimators, xMem decreases the median relative error by 84.32%, reduces the average
probability of estimation failure by 73.44%, accelerates the runtime by 50.16%, and improves memory conservation
by 125.36%.


Through this README, you can use xMem to estimate peak GPU memory for various deep learning models, run the entire
experiments to compare xMem against the baseline methods (DNNMem, SchedTune, and LLMem), and replicate the experimental
data presented in the figures and tables of the paper.

## Third-Party Code & Licensing
- The `exp/baselines/LLmem/` directory contains code from the [LLMem](https://github.com/taehokim20/LLMem) project. This code is subject to its original license, the full text of which is included in the `LICENSE` file within that directory.
- The `exp/baselines/LLmem/ColossalAI/` directory contains code from the [Colossal-AI](https://github.com/hpcaitech/ColossalAI/tree/v0.3.0) project. This code is subject to its original license, the full text of which is included in the `LICENSE` file within that directory.
- The `exp/baselines/schedtune/src` directory contains code from the [SchedTune](https://github.com/hadeelalbahar/SchedTune) project. This code is subject to its original license, the full text of which is included in the `LICENSE` file within that directory.


## 📂Project Structure
> [!Note]
> This project structure is temporary for prototype and will be refactored in the future.

```text
exp/                                    # Experiments related to xMem
    ├── baselines/                      # Baselines for xMem
    │   ├── LLMem/                      # LLMem Baseline
    │   ├── SchedTune/                  # SchedTune Baseline
    │   ├── dnnmem/                     # DNNMem Baseline
    │   └── solution/                   # a xMem class implementation for Experiments, following the same Interface as other baselines
    │── Experiments-ANOVA.ipynb         # Jupyter Notebook for ANOVA Experiment
    │── Experiments-Monte Carlo.ipynb   # Jupyter Notebook for Monte Carlo Experiment
    └── CoLab_large_Model.ipynb         # Jupyter Notebook for Large Model Experiment in CoLab
paper_container/                        # Base image building code
perf_estimator/                         # xMem Estimator code
    │── allocator/                      # Code for the Two-Layers Simulator
    │── profiler/                       # Code for the Analyzer
    │── estimator.py                    # Code for Orchestrator
    └── xmem.py                         # The entry point of xMem
plot/                                   # Code for plotting the results
xProfile.py                             # CPU-based profiler for generating the profiler file
main.py                                 # The CLI entry point for xMem
apps.py                                 # The CLI entry point for docker building and cleanup stuffs.
requirement.txt                         # Requirements for xMem
requirement-r.txt                       # Requirements for experiments
```

## ✅Compatibility

- ✅: Runnable
- ❌: Unrunnable
- ⚠️: Runnable with unverified results

| **Components** | **Linux (Debian 12 Recommended)** | **Windows 11** | **Mac Sequoia** |
| :------------: |:---------------------------------:|:--------------:|:---------------:|
|      xMem      |                 ✅                 |       ✅        |        ✅        |
|    xProfile    |                 ✅                 |       ⚠️       |       ⚠️        |
|  Plot Results  |                 ✅                 |       ✅        |        ✅        |
|  Experiments   |                 ✅                 |       ❌        |        ❌        |

## 🚧 Installation
> [!IMPORTANT]
> The installation instructions in this section is **Only** for xMem itself, not for Experiment.
>
> Please jump to [Experiments](#-installation-1) and follow the installation instructions under it
> if you want to run the experiments as experimental env is also work for xMem. Therefore, you can
> use that env to run both code (xMem itself and experiments).



### Miniconda (Optional)

[Miniconda](https://docs.anaconda.com/miniconda/) is recommended to manage the environment.
You can follow the below steps to create a new environment and install the required packages.

```shell
conda create -n xmem python=3.11 -y
```

```shell
conda activate xmem
```


### PyTorch

Please use the following command to install CPU-Only PyTorch

```shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

### 📦 Install Dependencies

```shell
pip install -r requirement.txt
```


## ⚙️ Usage


### Option 1 - (Recommanded) When you don't have profiling data, Using CPU-Based Profiler

> [!TIP]
> xProfile Tool blow could help you to generate the profiler file,
> or you can use the profiler file in `examples` folder via instructionms in below sections:
> - [Example 1](#quick-example---non-oom-example)
> - [Example 2 - OOM](#quick-example---oom-example)
> - [Example 3 - Transformer](#quick-example-for-transformer-model)


Ensure that you alreayd export the right PYTHONPATH. If not, run commands below.

After run, a webpage will also be opened in your browser, showing the memory usage curve of the model by plotly.

```shell
export PYTHONPATH="$(pwd)"
```


Please use 'xProfile.py' to generate the profiler file.

- The profiling file is a json file that contains the memory usage in the model.
- The profiling file is used as an input to the main.py to estimate the peak memory usage of the model.
- Ensure that you run a profiler job on Linux.

```shell
# python xProfile.py --help for more usage detail
python xProfile.py -m "VGG19" -b 130 -o "SGD"
```

> [!HIT]
> Both path of the profiling file and estimation command are shown in STDOUT, like below

Output is shown below
```text
Preparing VGG19 with fp16: False and optimiser: SGD
Loaded VGG19 in data type: torch.float32
Training on CPU Started
Initializing Training...
Training...
Using Convolutional Training loop.
Profiled data for VGG19 with batch size 130 and optimizer SGD
The file is saved to <path of result>
Command for Estimation: <------ a command can be run directly in shell, change max GPU memory before run
python main.py <path of profiling file> -b 130 -g <int: max mem in GB>

```

```shell
# python xProfile.py --help for more usage detail
python xProfile.py -m "facebook/opt-350m" -b 10 -o "AdamW"
```

Output is shown below
```text
Preparing facebook/opt-350m with fp16: False and optimiser: Adafactor
Loaded facebook/opt-350m in data type: torch.float32
Training on CPU Started
Initializing Training...
Training...
Using Mixed Precision (FP32) Training loop.
Profiled data for facebook/opt-350m with batch size 10 and optimizer Adafactor
The file is saved to <path of result>
Command for Estimation: <------ a command can be run directly in shell, change max GPU memory before run
python main.py <path of profiling file> -b 10 -m 'facebook/opt-350m' -i -g <int: max mem in GB>
```

There are only the below models supported for profiling

```text
CNN MODELS SUPPORTED:
        VGG16
        VGG19
        ResNet101
        ResNet152
        MobileNetV2
        MobeNetV3Small
        MobeNetV3Large
        MnasNet
        ConvNeXtTiny
        ConvNeXtBase
        RegNetX400MF
        RegNetY400MF


Optimizer for Transformer Models Supported:
        SGD
        Adam
        AdamW
        Adafactor


Optimizer for CNN Models Supported:
        SGD
        Adam
        RMSprop
        Adagrad
        AdamW

Transformer Models Supported:
    Technically, all the transformer models supported by HuggingFace are supported.
    However, the training loop and data loader (only wiki-text available) are not implemented for all the models.
```



### Option 2 - When you want to estimate your own profiling data

Ensure that you alreayd export the right PYTHONPATH. If not, run commands below.

After run, a webpage will also be opened in your browser, showing the memory usage curve of the model by plotly.

```shell
export PYTHONPATH="$(pwd)"
```

Command Usage

```text
NAME
    main.py

SYNOPSIS
    main.py PROFILER_FILE <flags>

POSITIONAL ARGUMENTS
    PROFILER_FILE
        Type: str

FLAGS
    -i, --is_transformer=IS_TRANSFORMER
        Type: bool
        Default: False
    -m, --model_name=MODEL_NAME
        Type: Optional[Optional]
        Default: None
    -b, --batch_size=BATCH_SIZE
        Type: int
        Default: 200
    -g, --gpu_memory_in_gb=GPU_MEMORY_IN_GB
        Type: Union
        Default: 8

NOTES
    PROFILER_FILE is a json file generated by xProfile.py.
    The profiling file is used as an input to the main.py to estimate the peak memory usage of the model.
    Ensure that you run a profiler job on Linux.
    Option -m is mandatory if option -i is set to True.
```

#### 🚀Quick Example - OOM Example

> [!WARNING]
> Failed to import pytorch `fbgemm.dll` warnning in Windows.
> Please follow the [FAQ](#windows---failed-to-import-pytorch-fbgemmdll-or-one-of-its-dependencies-is-missing) to solve the issue.

Run command below

```shell
## Guarantee that your inputs are same as the inputs which you used to profile the model
python main.py ./examples/convnext-base-batch130.json -b 130 -g 4
```

Result shows below:

```text
======================== Basic Information ========================
Batch Size: 130
Max GPU Memory: 4 GB
Runtime: 17205559489 s
======================== Estimated Result ========================
OOM: True  <---- This means that 4GB is not enough to run the model
4.00 GB is not enough to run the model
```

#### 🚀Quick Example - Non OOM Example

```shell
python main.py ./examples/convnext-base-batch130.json -b 130 -g 8
```

Result shows below:

```text
======================== Basic Information ========================
Batch Size: 130
Max GPU Memory: 8 GB
Runtime: 17227396749 s
======================== Estimated Result ========================
OOM: False
Estimated Peak GPU Memory: 5.22 GB
Estimated Peak Tensor Memory: 4.96 GB
Estimated result is saved in examples/xMem-result-convnext-base-batch130.json
```

#### 🚀Quick Example for Transformer Model

```shell
python main.py ./examples/facebook-opt-125m-batch34.json -b 34 -g 12 -m 'facebook/opt-125m' -i
```

Result shows below:
```text
======================== Basic Information ========================
Batch Size: 34
Max GPU Memory: 12 GB
Runtime: 13838115258 s
======================== Estimated Result ========================
OOM: False
Estimated Peak GPU Memory: 7.70 GB
Estimated Peak Tensor Memory: 6.86 GB
Estimated result is saved in examples/xMem-result-facebook-opt-125m-batch34.json

```



## 🧹(Optional) Clean Up
```shell
conda deactivate
```
```shell
conda env remove --name xmem
```


# ⚖️ Experiments

## ✅ Hardware Compatibility

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


## 🚧 Installation

### Miniconda
This project requires Miniconda for Python package management and Docker to run the evaluation experiments.
Please install them before proceeding:
- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
- [Docker](https://docs.docker.com/engine/install/)

First, create and activate a new Conda environment for this project:
```shell
conda create -n xmem-exp python=3.11 -y
```

```shell
conda activate xmem-exp
```

Next, install PyTorch. The following command installs the version used in our experiments (PyTorch 2.6.0 for CUDA 12.4).
```shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
For other CUDA versions, please refer to the [PyTorch 2.6.0 installation documentation](https://pytorch.org/get-started/previous-versions/).

### 📦 Install Dependencies
Finally, install the remaining Python packages using the provided requirements file:
```shell
pip install notebook # For Jupyter Notebook
pip install -r requirement-r.txt
```

### 💿Base Images

#### 1️⃣ Docker Pull Images

First, pull the base PyTorch images from Docker Hub, which are required to build the specific environments for xMem
and the baseline estimators. The exact image link for version 2.3.1 can be found [here](https://hub.docker.com/layers/pytorch/pytorch/2.3.1-cuda12.1-cudnn8-devel/images/sha256-a22a1fca37f8361c8a1e859cd6eb6bd9d1fb384f9c0dcb2cfc691a178eb03d17?context=explore)

```shell
docker pull pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```

```shell
docker pull pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
```

```shell
docker pull pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
```

#### 2️⃣ Build Runtime Images

> [!WARNING]
> Ensure you set the right PYTHONPATH before running the experiments. The PYTHONPATH should be the root directory of the project.

The LLMem baseline requires a custom Docker image due to specific dependencies. Build it using the following commands:

>[!TIP]
> You could also build environment yourself via this [link](https://github.com/taehokim20/LLMem)
```bash
cd exp/baselines/LLmem
docker build -t llmem .
cd ../../..
```

After build, you will get a base image, called `llmem`.
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

After preparing the base images, run the provided script to automatically build the remaining Docker images
for the experiments:
```shell
python apps.py experiments prepare
```

#### 3️⃣ Check runtime images

check whether both image are built successfully

```shell
docker images
```

shown as
```text
repository              tag                           image id       created         size
llmem-estimator         latest                        100b183a6cc8   2 minutes ago   13.8gb
schedtun-estimator      latest                        006e83a1937f   2 minutes ago   18.7gb
paper-experiments       latest                        8ae7c8a4b68c   2 minutes ago   18.2gb
paper-estimator         latest                        ebe543382d65   4 minutes ago   13.9gb
paper-experiments-llm   latest                        0eadd4567983   4 minutes ago   13.8gb
llmem                   latest                        1af477f97e32   9 minutes ago   13.7gb
pytorch/pytorch         2.6.0-cuda12.4-cudnn9-devel   7d57e307bd9c   3 months ago    13.2gb
pytorch/pytorch         2.3.1-cuda12.1-cudnn8-devel   b40b101922fd   11 months ago   17.1gb
pytorch/pytorch         2.0.1-cuda11.7-cudnn8-devel   42a0e9b621e2   2 years ago     13.2gb
```

## 🚀 Execute Experiments

> [!WARNING]
> Do not run any GPU-related tasks on the GPUs used during the experiment, as they will be occupied for specific purposes.

> [!WARNING]
> Multiple directories are created by experiments:
> - `~/CNN-Exp` to store result related to CNN models
> - `~/Transformer-Exp` to store result related to Transformer Models
> - `~/Large-Transformer-Exp` to store result related to Qwen3 0.6B and Pythia 1B

### Run ANOVA Experiment
> [!IMPORTANT]
> Ensure that you have right configuration of interpreter for Jupyter environment.
> Additionally, the entire ANOVA experiment generally take more than a week time to run.

> [!Caution]
> The result can be only valid and visualized when entire experiment are completely finished,
> due to all samples will be run once with xMem and other baselines.

> [!Caution]
> Please note, the ANOVA experiment will create more than thousands docker instants each times.

Notions of variable in this notebook:
- `conf_index`: index of the test configuration (value: 0-2), shown as below
  - `0`: CNN models. The data is used for Research Question 1-4 in the Paper.
  - `1`: Transformer models. The data is used for Research Question 1-4 in the Paper.
  - `2`: Larger Transformer models. The data is used for Research Question 5 in the Paper.
- `config.repeats = 5`: how many times does each test configuration run repeatedly. Default is 5
- `config.gpu_id = 0`: Index of GPU
- `config.result_verification = True`: do not change, as it enables estimated memory verification on the GPU.
- `config.debug = False`: do not change


Use a Jupyter [Notebook](exp/Experiments-ANOVA.ipynb) for this experiments


### Run Monte Carlo Experiment
> [!IMPORTANT]
> Ensure that you have right configuration of interpreter for Jupyter environment

> [!Caution]
> Please note, the Monte Carlo experiment will create amount of docker instants, which equal
> to the number of 'total_run' in notebook.
>
> Moreover, we used two GPUs in this experiment, so the default value of variable `gpu_ids` is `[0, 1]`.
> You can change this list.

Notions of variable in this notebook:
- **`total_run`**: The total number of random configurations to sample and run. A minimum of 20 is recommended.
- **`gpu_ids`**: A list of GPU device indices (e.g., `[0, 1]`) from which the experiment will randomly select a GPU for each run.
- **`config.repeats`**: Must remain `1` for Monte Carlo experiments since the given random test configuration should be only run once.
- **`config.gpu_id`**: This value is ignored and will be overwritten by a randomly selected index from `gpu_ids` for each run.
- **`config.result_verification`**: do not change, as it enables estimated memory verification on the GPU.

Use a Jupyter [Notebook](exp/Experiments-Mento%20Carlo.ipynb) for this experiments

## 🧹(Optional) Clean Up

### Docker Instances
> [!IMPORTANT]
> - The command will remove all stopped containers and all dangling images
> - Please do not execute this command if you have concerns about the code, as it involves `delete` operations in level of docker.
```shell
python app.py cleanup
```

### Conda Environment

```shell
conda deactivate
```
```shell
conda env remove --name xmem-exp
```


# 📊 Plot the Results

> [!WARNING]
> Ensure that you have already installed the PyTorch following the above [steps](#pytorch)

The instructions for plotting the figures presented in the paper (e.g., Figures 7, 8, and 9) from the raw
experimental data are located in a separate directory.

To reproduce all figures, please follow the detailed steps provided in
this guide: [here](plot/README.md).

```shell
pip install notebook # For Jupyter Notebook
pip install -r requirement-r.txt
```

# ❓ FAQ

## Windows - Failed to import pytorch `fbgemm.dll` or one of its dependencies is missing

Solution is that download `Visual C++ Redistributable for Visual Studio 2019` from Microsoft official website and install it.
The download linke is [here](https://my.visualstudio.com/Downloads?q=c++%20redistributable)

## Insufficient RAM

Solution is Swap. You could create a large Swap file for your linux instead of RAM. The link below may helps you setting Swap file.
- https://linuxize.com/post/create-a-linux-swap-file/


# ✅ To-Do List
- [ ] Refactor the project structure
- [ ] Optimization of Analyzer for runtime performance
- [ ] Handle a new memory allocation event when the previous memory remains in use because the deallocation event between these two moments is missing.



# 📚 Resources
- [SchedTune Source Code](https://github.com/hadeelalbahar/SchedTune)
- [LLMem Source Code](https://github.com/taehokim20/LLMem)
- [ColossalAI-Version 0.3.0](https://github.com/hpcaitech/ColossalAI/tree/v0.3.0)
