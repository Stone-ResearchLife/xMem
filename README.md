# 🧮 XMem: A Cross-Architecture GPU Memory Estimator


xMem is a novel framework designed to accurately estimate the peak GPU
memory consumption for DL training jobs using data from a profiler, a
performance analyzer frequently used during DL model development.

Its accurate GPU memory estimation is essential to mitigate these costly
OOM failures, optimize scheduling, and improve the overall cluster efficiency
and stability. Specifically, precise GPU memory estimation for DL jobs enables
decision systems within shared GPU clusters to implement more effective intelligent resource
scheduling, leading to substantial GPU memory conservation and optimized asset
utilization, which in turn helps mitigate the prevailing GPU scarcity.

## 📂Project Structure
> [!NOTE] This project structure is temporary for prototype and will be refactored in the future.

```text
exp/                                    # Experiments related to xMem
    ├── baselines/                      # Baselines for xMem
    │   ├── LLMem/                      # LLMem Baseline
    │   ├── SchedTune/                  # SchedTune Baseline
    │   ├── dnnmem/                     # DNNMem Baseline
    │   └── solution/                   # a xMem class implementation for Experiments, following the same Interface as other baselines
    │── Experiments-ANOVA.ipynb         # Jupyter Notebook for ANOVA Experiment
    │── Experiments-Mento Carlo.ipynb   # Jupyter Notebook for Monte Carlo Experiment
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
> Please jump to [Experiments](#-execute-experiments) if you want to run the experiments directly.


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

## 📏 (Optional) CPU-Based Profiler

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

```shell
# python xProfile.py --help for more usage detail
python xProfile.py -m "facebook/opt-350m" -b 10 -o "AdamW"
```

There are only the below models supported for profiling

```text
CNN MODELS SUPPORTED:
        VGG16
        VGG19
        ResNet101
        esNet152
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
    However, the training loop and data loader are not implemented for all the models.
```

The path of profiling data JSON file will be shown in the last line of stdout, like
```text
Preparing facebook/opt-350m with fp16: False and optimiser: AdamW
Loaded facebook/opt-350m in data type: torch.float32
Training on CPU Started
Initializing Training...
Training...
Profiled data for facebook/opt-350m with batch size 10 and optimizer AdamW
The file is saved to ~/DL-Estimator/20250529-161258-7fbd/results  <---- Path of file shows here
```

## ⚙️ Usage

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

### 🚀Quick Example - OOM Example

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

### 🚀Quick Example - Non OOM Example

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

### 🚀Quick Example for Transformer Model

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


# ⚖️ Execute Experiments

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

### Miniconda (Optional)

[Miniconda](https://docs.anaconda.com/miniconda/) is recommended to manage the environment.
You can follow the below steps to create a new environment and install the required packages.

```shell
conda create -n xmem-exp python=3.11 -y
```

```shell
conda activate xmem-exp
```

The below command, as an example, was used in experiments with the local environment (CUDA 12.4).
```shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
You can also see official document of PyTorch 2.6.0 installation [here](https://pytorch.org/get-started/previous-versions/)

### 📦 Install Dependencies

```shell
pip install notebook # For Jupyter Notebook
pip install -r requirement-r.txt
```

### 💿Base Images
Ensure Docker Client is installed on your machine. If not follow the instructions [here](https://docs.docker.com/engine/install/).

#### 1️⃣ Docker Pull Images

Pull the base image from Docker Hub. This image is used to build images for the experiments.
Image Link: [here](https://hub.docker.com/layers/pytorch/pytorch/2.3.1-cuda12.1-cudnn8-devel/images/sha256-a22a1fca37f8361c8a1e859cd6eb6bd9d1fb384f9c0dcb2cfc691a178eb03d17?context=explore)

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

Since a special build requirement of LLMem, we have to build base image for LLMem by

>[!TIP]
> You could also build environment yourself via this [link](https://github.com/taehokim20/LLMem)
```bash
cd exp/baselines/LLmem
docker build -t llmem .
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
> [!IMPORTANT] Ensure that you have right configuration of interpreter for Jupyter environment

Use a Jupyter [Notebook](exp/Experiments-ANOVA.ipynb) for this experiments


### Run Monte Carlo Experiment
> [!IMPORTANT] Ensure that you have right configuration of interpreter for Jupyter environment
 
Use a Jupyter [Notebook](exp/Experiments-Mento%20Carlo.ipynb) for this experiments

## 🧹(Optional) Clean Up

### Docker Instances
> [!IMPORTANT]
> - The command will remove all stopped containers and all dangling images
> - Please do not execute this command if you have concerns about the code, as it involves a `delete` operation.
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

Install the required packages and
follow [steps](plot/README.md) to generate all the plots mentioned in the paper.

```shell
pip install notebook # For Jupyter Notebook
pip install -r requirement-r.txt
```

# ❓ FAQ

## Windows - Failed to import pytorch `fbgemm.dll` or one of its dependencies is missing

Solution is that download `Visual C++ Redistributable for Visual Studio 2019` from Microsoft official website and install it.
The download linke is [here](https://my.visualstudio.com/Downloads?q=c++%20redistributable)


# ✅ To-Do List
- [ ] Refactor the project structure
- [ ] Optimization of Analyzer for runtime performance
- [ ] Handle a new memory allocation event when the previous memory remains in use because the deallocation event between these two moments is missing.



# 📚 Resources
- [SchedTune Source Code](https://github.com/hadeelalbahar/SchedTune)
- [LLMem Source Code](https://github.com/taehokim20/LLMem)
