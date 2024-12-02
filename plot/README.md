# Experimental Plots

## 1️⃣ Dependencies
Ensure all dependencies are installed, as shown in the [README](../README.md).

# 2️⃣ Data Structure
The result data is consist of three parts:
- `train info`: contains all training parameters, such as batch size, model name, optimizer, etc.
- `config`: contains all basic information realted to this run.
- `solution`: denotes the evaluation result by xMem, termed as `xMem (this paper)`
- `dnnmem`: denotes the evaluation result by DNNMem, termed as `DNNMem`
- `schedtune`: denotes the evaluation result by SchedTune, termed as `SchedTune`
- `llmem`: denotes the evaluation result by LLMem, termed as `LLMem`

## Evaluation Result Structure
|      Field in JSON       |   Name in Dataframe   |            Symbol             |                                                                           Description                                                                            |
|:------------------------:|:---------------------:|:-----------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|        `runtime`         |       `runtime`       |             None              |                                                               The actual execution time in second                                                                |
|         `memory`         |       `memory`        | $\hat{M}^{\text{peak}}_{jde}$ |                                                  The peak memory usage as predicted by estimator, as expressed                                                   |
|          `oom`           |         `oom`         |   $\hat{\text{OOM}}_{jde}$    |                                                        Boolean prediction of OOM occurrence by estimator                                                         |
|         `ground`         |       `ground`        |    $M^{\text{peak}}_{jd1}$    |                                                            The peak memory usage as recorded by NVML                                                             |
|         `error`          |        `error`        |     $\text{error}_{jde1}$     |                            The relative error of $M^{\text{peak}}_{jd1}$ relative to $\hat{M}^{\text{peak}}_{jde}$ for 1st validation                            |
|        `real_oom`        |      `real_oom`       |      $\text{OOM}_{jd1}$       |                                           Boolean indicating actual OOM occurrence during training for 1st validation                                            |
|   `correct_estimation`   | `correct_estimation`  |          $C_{jde1}$           |                      Boolean indicating if the prediction $\hat{\text{OOM}}_{jde}$ matches the actual $\text{OOM}_{jd1}$ for 1st validation                      |
|    `2nd verification`    |        `None`         |             None              |                                          The Field is only available when $C_{jde1}=True \land \text{OOM}_{jd1}=False$                                           |
|  `2nd verification.oom`  |    `2nd_real_oom`     |      $\text{OOM}_{jd2}$       |                                           Boolean indicating actual OOM occurrence during training for 2nd validation                                            |
| `2nd verification.error` |      `2nd_error`      |     $\text{error}_{jde2}$     | The relative error of $M^{\text{peak}}_{jd2}$ relative to $\hat{M}^{\text{peak}}_{jde}$ for 2nd validation. It set to `null` when `2nd verification.oom` is True |
|          `None`          | `accurate_estimation` |          $C_{jde2}$           |                                                     Boolean indicating if the prediction  for 1st validation                                                     |
|          `None`          |    `save_memory`      |   $M^{\text{save}}_{jde}$     |                                                    The memory conserved by estimator                                                                             |


# 3️⃣ Plotting
## Using Jupyter Notebook
We provided a Jupyter notebook in the respority to plot the experimental results. You can find the notebook in the `plot` directory, named as `Figures in Paper.ipynb`
## Using Python Code Snippet
```python
from pathlib import Path
from plot.plot import ExperimentPlot

# There are data directories for each experiment
# 1. ANOVA Experments: <project base>/plot/data/ANOVA
# 2. Monte Carlo: <project base>/plot/data/MonteCarlo
data_dir = Path("<Path of data dir>")
output_dir = Path("<Directory for saving plots>") 
output_dir.mkdir(exist_ok=True, parents=True)

# Initialize the ExperimentPlot object
e_plot = ExperimentPlot(output_dir=output_dir)

# plot a relative error box diagram as an exmaple
fig = e_plot.plot_relative_error_in_box_diagram_with_verification_data(
    title="Evaluation-Relative Error Experiments Result across Estimators-SGD",
    data_dir=data_dir,
    overall_median=False, # show overall median relative error near the legend box, only working for image size (2000, 450)
    image_size=(2000, 450), # figure size
    font_size=20, # x and y label font size
    legend_font_size=18, # legend font size
    tickfont_size=22, # tick font size
    view_mode = True # default setting for easily viewing the plot in notebook. all parameters, like image_size, font_size, legend_font_size, tickfont_size, will be ignored
)
fig.show()
```

# 4️⃣ Output
The figure in PDF format will be saved into your output folder with the name you typed in the `title` parameter.