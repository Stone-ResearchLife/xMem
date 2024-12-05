# 🧮 XMem: A Cross-Architecture GPU Memory Estimator

xMem, a GPU memory estimator that can utilize CPU-based profiling data to infer the peak 
memory required for training a model on a GPU

## 🚧 Pre-requisite 
```shell
pip install -r requirement.txt
```

## ⚙️ CMD Usage
Ensure that you alreayd export the right PYTHONPATH. If not, run below command
```shell
export PYTHONPATH="$(pwd)"
```
Command Usage
```text
NAME
    main.py

SYNOPSIS
    main.py MODEL PROFILER_FILE <flags>

POSITIONAL ARGUMENTS
    MODEL
        Type: str
    PROFILER_FILE
        Type: str

FLAGS
    -b, --batch_size=
        Type: int
        Default: 200
    -i, --input_size=INPUT_SIZE
        Type: int
        Default: 86
    -g, --gpu_memory_in_gb=MAX_GPU_MEMORY_IN_GB
        Type: Union
        Default: 4


```
### 🚀Quick Example - OOM Example
Run command below
```shell
## Guarantee that your idnputs are same as the inputs which you used to profile the model
python main.py "ConvNeXtBase" ./examples/convnext-base-batch130.json -b 130 -g 4

```
Result shows below:
```text
======================== Basic Information ========================
Model: ConvNeXt
Batch Size: 130
Input Size: [3, 86, 86]
Max GPU Memory: 4GB
======================== Estimated Result ========================
OOM: True      <---- This means that 4GB is not enough to run the model
4294967296GB is not enough to run the model
```
The Last Frame of Memory Snapshot will be saved in the log folder. The file name will be `Last-frame.png`.
The Image is only for debugging purpose and is shown [here](docs/Last-frame.png)

### 🚀Quick Example - Non OOM Example
```shell
python main.py "ConvNeXtBase" ./examples/convnext-base-batch130.json -b 130 -g 8
```
Result shows below:
```text
======================== Basic Information ========================
Model: ConvNeXt
Batch Size: 130
Input Size: [3, 86, 86]
Max GPU Memory: 8GB
======================== Estimated Result ========================
OOM: False
Estimated Peak GPU Memory: 5.26GB    <---- This is the peak memory that the model will use in the GPU
Estimated Peak Tensor Memory: 4.99GB
```

## 📏 CPU-Based Profiler
Please use 'xmem_profile.py' to generate the profiler file. 
The profiler file is a json file that contains the memory usage in the model. 
The profiler file is used as an input to the main.py to estimate the peak memory usage of the model.

```shell
# python profile.py --help for more usage detail
python xProfile.py -m "VGG19" -b 100
```

There are only the below models supported for profiling
```text
MODELS SUPPORTED:
        VGG11
        VGG16
        VGG19
        ResNet50
        ResNet101
        esNet152
        MobileNetV2
        MobeNetV3Small
        MobeNetV3Large
        MnasNet
        ConvNeXtTiny
        ConvNeXtBase
        RegNetX400MF
        RegNetX32GF
        RegNetY400MF
        RegNetY32GF
```

# 📊 Plot the Results
Install the required packages and 
follow [steps](plot/README.md) to generate all the plots mentioned in the paper.

```shell
pip install -r requirement-r.txt
```

# ⚖️ Execute Experiments
Install the required packages and read [here](experiments/README.md) for more details
```shell
pip install -r requirement-r.txt
```
