# 🧮 XMem: A Cross-Architecture GPU Memory Estimator

xMem, a novel tool that uses CPU-based analysis to accurately predict
the memory required for model training on a GPU. By eliminating the
reliance on GPUs for memory estimation, xMem promotes efficient
GPU utilization while mitigating OOM errors.

## ✅Compatibility

- ✅: Runnable
- ❌: Unrunnable
- ⚠️: Runnable with unverified results

Please check [README](experiments/README.md) for Hardware Compatibility

| **Components** | **Linux (Recommended)** | **Windows** | **Mac** |
| :------------: | :---------------------: | :---------: | :-----: |
|      xMem      |           ✅            |     ✅      |   ✅    |
|    xProfile    |           ✅            |     ⚠️      |   ⚠️    |
|  Plot Results  |           ✅            |     ✅      |   ✅    |
|  Experiments   |           ✅            |     ❌      |   ❌    |

## 🚧 Pre-requisite

### Miniconda (Optional)

[Miniconda](https://docs.anaconda.com/miniconda/) is recommended to manage the environment.
You can follow the below steps to create a new environment and install the required packages.

```shell
conda create -n xmem python=3.10 -y
```

```shell
conda activate xmem
```

### PyTorch

Please use the following command to install CPU-Only PyTorch

```shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Or You cloud use the below command to install the GPU version based on your CUDA version.
The below command is for CUDA 12.4

```shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

You can also see official document of PyTorch 2.3.0 installation [here](https://pytorch.org/get-started/previous-versions/#v230)

### Other Dependencies

```shell
pip install -r requirement.txt
```

## ⚙️ CMD Usage

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
Input Size: [3, 86, 86]
Max GPU Memory: 4 GB
Runtime: 39.01 s
======================== Estimated Result ========================
OOM: True      <---- This means that 4GB is not enough to run the model
4.00 GB is not enough to run the model
```

The Last Frame of Memory Snapshot will be saved in the log folder. The file name will be `Last-frame.png`.
The Image is only for debugging purpose and is shown [here](docs/Last-frame.png)

### 🚀Quick Example - Non OOM Example

```shell
python main.py ./examples/convnext-base-batch130.json -b 130 -g 8
```

Result shows below:

```text
======================== Basic Information ========================
Batch Size: 130
Input Size: [3, 86, 86]
Max GPU Memory: 8 GB
Runtime: 37.01 s
======================== Estimated Result ========================
OOM: False
Estimated Peak GPU Memory: 5.26GB    <---- This is the peak memory that the model will use in the GPU
Estimated Peak Tensor Memory: 4.99GB
```

### 🚀Quick Example for Transformer Model

```shell
python main.py ./examples/facebook-opt-125m-batch34.json -b 34 -g 12 -m 'facebook/opt-125m' -i
```


## 📏 CPU-Based Profiler

Please use 'xmem_profile.py' to generate the profiler file.

- The profiling file is a json file that contains the memory usage in the model.
- The profiling file is used as an input to the main.py to estimate the peak memory usage of the model.
- Ensure that you run a profiler job on Linux.

```shell
# python xProfile.py --help for more usage detail
python xProfile.py -m "VGG19" -b 130 -o "SGD"
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

Optimizer Supported:
        SGD
        Adam
        RMSprop
        Adagrad
        AdamW
        Adafactor

Transformer Models Supported:
    Technically, all the transformer models supported by HuggingFace are supported.
    However, the training loop and data loader are not implemented for all the models.
```

# 📊 Plot the Results

> [!WARNING]
> Ensure that you have already installed the PyTorch following the above [steps](#pytorch)

Install the required packages and
follow [steps](plot/README.md) to generate all the plots mentioned in the paper.

```shell
pip install -r requirement-r.txt
```

# ⚖️ Execute Experiments

> [!WARNING]
> Ensure that you have already installed the PyTorch following the above [steps](#pytorch)

Install the required packages and read [here](experiments/README.md) for more details

```shell
pip install -r requirement-r.txt
```

# ❓ FAQ

## Windows - Failed to import pytorch `fbgemm.dll` or one of its dependencies is missing

Solution is that download `Visual C++ Redistributable for Visual Studio 2019` from Microsoft official website and install it.
The download linke is [here](https://my.visualstudio.com/Downloads?q=c++%20redistributable)
