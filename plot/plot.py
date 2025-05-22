import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import copy
import numpy as np
from enum import Enum
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Union, Dict, Optional
from perf_estimator.utilis import filter_files
from perf_estimator.utilis.utilis import temp_dir_with_specific_path


class ApproachedName(Enum):
    solution = "xMem (this paper)"
    DNNmem = "DNNMem"
    SchedTune = "SchedTune"
    LLmem = "LLMem"


class ExperimentPlot:
    def __init__(self, output_dir: Union[str, Path] = None):
        # initialize the work dir
        self._work_dir = output_dir or temp_dir_with_specific_path("xMem-plot")
        self._work_dir = Path(self._work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)

        self._color_scheme = {
            ApproachedName.solution.value: "#377eb8",
            ApproachedName.DNNmem.value: "#e41a1c",
            ApproachedName.SchedTune.value: "#4daf4a",
            ApproachedName.LLmem.value: "#984ea3",
        }
        self._name_map = {
            "solution": ApproachedName.solution.value,
            "DNNmem": ApproachedName.DNNmem.value,
            "SchedTune": ApproachedName.SchedTune.value,
            "LLmem": ApproachedName.LLmem.value,
        }
        self.font = dict(size=30)
        self.legend_font = dict(size=25)
        self._plotly_template = "plotly_white"
        self._output_format = "pdf"
        self._line_width = 6  # for two-columns paper
        self._tickfont_size = 25
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_image_dir(self, title: str) -> Path:
        file_name = f"{title}.{self._output_format}"
        _dir = self._work_dir.joinpath("images")
        if _dir.exists() is False:
            _dir.mkdir(parents=True, exist_ok=True)
        return _dir.joinpath(file_name)

    def data_processing(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        if str(data_dir) not in self._cache.keys():
            evaluates_files = filter_files("evaluation_result.json", data_dir, False)
            random_result = []
            for eva_file in evaluates_files:
                with open(eva_file, "r") as f:
                    eva_data = json.load(f)
                basic_info = eva_data["train info"]
                basic_info["timestamp"] = "-".join(
                    str(eva_data["config"]["run_id"]).split("-")[:2]
                )
                del eva_data["train info"], eva_data["config"]
                for key, value in eva_data.items():
                    copy_basic_info: dict = copy.deepcopy(basic_info)
                    copy_basic_info["tool"] = key
                    copy_basic_info["2nd_real_oom"] = value.get(
                        "2nd verification", {}
                    ).get("oom")
                    copy_basic_info["2nd_error"] = value.get(
                        "2nd verification", {}
                    ).get("error", value["error"])
                    if (
                        value["real_oom"] == False
                        and value["correct_estimation"] == True
                        and copy_basic_info["2nd_real_oom"] == False
                    ):
                        # The condition of C_moe2 = 1 in the Paper
                        accurate_estimation = True
                        save_memory = copy_basic_info["total_gpu_memory"] - (
                            value["memory"] / 1024**3
                        )
                    elif (
                        value["real_oom"] == True
                        and value["correct_estimation"] == True
                    ):
                        # The condition of C_moe2 = 1 or C_moe1 = 1 in the Paper
                        accurate_estimation = True
                        save_memory = copy_basic_info["total_gpu_memory"]
                    else:
                        # otherwise
                        accurate_estimation = False
                        save_memory = 0

                    copy_basic_info["accurate_estimation"] = accurate_estimation
                    copy_basic_info["save_memory"] = save_memory
                    del value["2nd verification"]
                    copy_basic_info.update(value)
                    random_result.append(copy_basic_info)
            df = pd.DataFrame(random_result)
            df["tool"] = df["tool"].replace(self._name_map)
            df["color"] = df["tool"].replace(self._color_scheme)
            self._cache[str(data_dir)] = df
        return self._cache[str(data_dir)].copy()

    def _add_four_quadrant(
        self, fig, quadrant_thresholds: tuple, max_y: int, font_size: int = 45
    ):
        # four-quadrant
        x_threshold = quadrant_thresholds[0]
        y_threshold = quadrant_thresholds[1]
        quadrant_scheme = {
            "1st": {
                # Bottom Left
                "fill": "rgba(44, 123, 182, 0.2)",
                "font": {"size": font_size, "color": "blue"},
            },
            "2nd": {
                # Bottom Right
                "fill": "rgba(253, 174, 97, 0.2)",
                "font": {"size": font_size, "color": "DarkOrange"},
            },
            "3rd": {
                # Top Left
                "fill": "rgba(171, 217, 233, 0.2)",
                "font": {"size": font_size, "color": "DarkTurquoise"},
            },
            "4th": {
                # Top Right
                "fill": "rgba(215, 25, 28, 0.2)",
                "font": {"size": font_size, "color": "red"},
            },
        }
        # 1st Quadrant: Bottom Left
        fig.add_shape(
            type="rect",
            x0=0,
            x1=x_threshold,
            y0=0,
            y1=y_threshold,
            fillcolor=quadrant_scheme["1st"]["fill"],
            line=dict(width=0),
        )
        fig.add_annotation(
            x=x_threshold / 2,
            y=y_threshold / 2,
            text="Optimal",
            showarrow=False,
            font=quadrant_scheme["1st"]["font"],
            xanchor="center",
            yanchor="middle",
            textangle=270,
        )
        # 2nd Quadrant: Bottom Right
        fig.add_shape(
            type="rect",
            x0=x_threshold,
            x1=100,
            y0=0,
            y1=y_threshold,
            fillcolor=quadrant_scheme["2nd"]["fill"],
            line=dict(width=0),
        )
        fig.add_annotation(
            x=(x_threshold + 100) / 2,
            y=y_threshold / 2,
            text="Underestimation",
            showarrow=False,
            font=quadrant_scheme["2nd"]["font"],
            xanchor="center",
            yanchor="middle",
        )
        # 3rd Quadrant: Top Left
        fig.add_shape(
            type="rect",
            x0=0,
            x1=x_threshold,
            y0=y_threshold,
            y1=max_y,
            fillcolor=quadrant_scheme["3rd"]["fill"],
            line=dict(width=0),
        )
        fig.add_annotation(
            x=x_threshold / 2,
            y=(y_threshold + 80) / 2,
            text="Overestimation",
            showarrow=False,
            font=quadrant_scheme["3rd"]["font"],
            xanchor="center",
            yanchor="middle",
            textangle=270,
        )
        # 4th Quadrant: Top Right
        fig.add_shape(
            type="rect",
            x0=x_threshold,
            x1=100,
            y0=y_threshold,
            y1=max_y,
            fillcolor=quadrant_scheme["4th"]["fill"],
            line=dict(width=0),
        )
        fig.add_annotation(
            x=(x_threshold + 100) / 2,
            y=(y_threshold + 80) / 2,
            text="Worst",
            showarrow=False,
            font=quadrant_scheme["4th"]["font"],
            xanchor="center",
            yanchor="middle",
        )

        # Add the threshold lines
        fig.add_shape(
            type="line",
            x0=x_threshold,
            x1=x_threshold,
            y0=0,
            y1=max_y,
            line=dict(color="black", width=2, dash="dash"),
        )

        fig.add_shape(
            type="line",
            x0=0,
            x1=100,
            y0=y_threshold,
            y1=y_threshold,
            line=dict(color="black", width=2, dash="dash"),
        )
        return fig

    def plot_memory_fluctuation_by_changing_zero_out(
        self,
        title,
        data_dir: Union[str, Path],
        image_size=(1500, 500),
        font_size=30,
        legend_font_size=25,
        tickfont_size=30,
        line_width=6,
        view_mode: bool = False,
    ):
        if view_mode:
            image_size = (1000, 400)
            font_size = 20
            legend_font_size = 15
            tickfont_size = 20
            line_width = 4

        self.font.update(dict(size=font_size))
        self.legend_font.update(dict(size=legend_font_size))
        from exp.snapshot import SnapshotAnalyser

        pickles = filter_files(".pickle", data_dir, fuzz=True)
        outputs = {}

        for index, p in enumerate(pickles):
            p = Path(p)
            model_name = p.parent.name.split("-")[-1]
            cate_data = p.name.split(".")[0]

            snap = SnapshotAnalyser(p)
            segs, traces = snap.fetch_gpu_segment_max_changes_directly()

            if model_name not in outputs.keys():
                outputs[model_name] = {}
            outputs[model_name][cate_data] = {
                "trace": [trace / 1024**3 for trace in traces],
                "segment": [seg / 1024**3 for seg in segs],
            }

        num_plots = len(outputs.keys())
        cols = min(num_plots, 3)
        rows = 1 if num_plots <= 3 else ((num_plots - 1) // 3) + 1

        fig = make_subplots(
            rows=rows, cols=cols, shared_xaxes=True, vertical_spacing=0.1
        )

        for index, (name, item) in enumerate(outputs.items()):
            col = (index % 3) + 1
            row = (index // 3) + 1
            beforeBW_data = item["beforeBW"]
            normal_data = item["normal"]
            showlegend = True if index == 0 else False

            # Point 0
            fig.add_trace(
                go.Scatter(
                    y=beforeBW_data["segment"],
                    mode="lines",
                    name="Segment (POS0)",
                    line=dict(color="Lime", width=line_width),
                    showlegend=showlegend,
                ),
                row=row,
                col=col,
            )

            # trace - Point 0
            fig.add_trace(
                go.Scatter(
                    y=beforeBW_data["trace"],
                    mode="lines",
                    name="Tensor (POS0)",
                    line=dict(color="Lime"),
                    showlegend=showlegend,
                    fill="tozeroy",
                ),
                row=row,
                col=col,
            )
            # Point 1
            fig.add_trace(
                go.Scatter(
                    y=normal_data["segment"],
                    mode="lines",
                    name="Segment (POS1)",
                    line=dict(color="Red", width=line_width, dash="dash"),
                    showlegend=showlegend,
                ),
                row=row,
                col=col,
            )
            # trace - Point 1
            fig.add_trace(
                go.Scatter(
                    y=normal_data["trace"],
                    mode="lines",
                    name="Tensor (POS1)",
                    line=dict(color="Red"),
                    showlegend=showlegend,
                    fill="tozeroy",
                ),
                row=row,
                col=col,
            )

            fig.update_xaxes(
                title_text=name,
                row=row,
                col=col,
                titlefont=self.font,
                tickfont=dict(size=tickfont_size),
            )
            fig.update_yaxes(
                title_text="Memory (GB)",
                row=row,
                col=col,
                titlefont=self.font,
                tickfont=dict(size=tickfont_size),
            )

        default_width = image_size[0]
        default_height = image_size[1] * rows
        fig.update_layout(
            height=default_height,
            width=default_width,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255, 0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font,
            ),
            showlegend=True,
            template=self._plotly_template,
            margin=dict(l=50, r=10, t=20, b=50, pad=2),
        )
        if view_mode is False:
            fig.write_image(
                file=self.get_image_dir(title),
                width=default_width,
                height=default_height,
            )
        return fig

    def _compare_snapshot_and_allocator_simulator(self, data_analysis, allocator):
        _sim_trace_record = allocator._trace._trace_history
        _real_trace_record = data_analysis.device_traces
        assert len(_sim_trace_record) != _real_trace_record
        memory_table = {"simulate": [], "real": []}
        keywords = ["segment_alloc", "segment_free"]
        sim_max = 0
        real_max = 0
        for _sim_trace, _real_trace in zip(_sim_trace_record, _real_trace_record):
            if _sim_trace["action"] == keywords[0]:
                sim_max += _sim_trace["size"]
            elif _sim_trace["action"] == keywords[1]:
                sim_max -= _sim_trace["size"]

            if _real_trace["action"] == keywords[0]:
                real_max += _real_trace["size"]
            elif _real_trace["action"] == keywords[1]:
                real_max -= _real_trace["size"]

            memory_table["simulate"].append(sim_max)
            memory_table["real"].append(real_max)
        return memory_table

    def plot_assessment_result_between_snapshot_and_simulation(
        self,
        title: str,
        data_dir: Union[str, Path],
        image_size=(1000, 500),
        tickfont_size=25,
        font_size=20,
        legend_font_size=16,
        max_gpu_memory_in_gb=7.6,
        view_mode: bool = False,
    ):
        if view_mode:
            image_size = (1000, 400)
            font_size = 15
            legend_font_size = 15
            tickfont_size = 15

        self.font.update(dict(size=font_size))
        self.legend_font.update(dict(size=legend_font_size))
        if str(data_dir) not in self._cache.keys():
            from perf_estimator.allocator import AllocatorSim
            from exp.snapshot import SnapshotAnalyser

            max_gpu_memory_in_gb = max_gpu_memory_in_gb
            max_sample = 23000
            pickles = filter_files(".pickle", data_dir, fuzz=True)
            outputs = {}
            for index, p in enumerate(pickles):

                _sim = AllocatorSim(max_allocated_memory_gb=max_gpu_memory_in_gb)
                p = Path(p)
                model_name = p.name.split(".")[0]
                snapshot_analyser = SnapshotAnalyser(str(p))
                memory_blocks = []
                for blocks in snapshot_analyser.activity_blocks.values():
                    for block in blocks:
                        memory_blocks.append(block)

                _result_allocate = _sim.simulate(memory_blocks, segment_plot=False)
                memory_data = self._compare_snapshot_and_allocator_simulator(
                    snapshot_analyser, _result_allocate
                )
                # Convert to GB
                memory_data["simulate"] = [
                    mem / 1024**3 for mem in memory_data["simulate"][0:max_sample]
                ]
                memory_data["real"] = [
                    mem / 1024**3 for mem in memory_data["real"][0:max_sample]
                ]
                outputs[model_name] = memory_data
            self._cache[str(data_dir)] = outputs

        outputs = self._cache[str(data_dir)]
        num_plots = len(outputs.keys())
        cols = min(num_plots, 3)
        rows = 1 if num_plots <= 3 else ((num_plots - 1) // 3) + 1

        fig = make_subplots(
            rows=rows, cols=cols, shared_xaxes=True, vertical_spacing=0.1
        )

        for index, (name, item) in enumerate(outputs.items()):
            col = (index % 3) + 1
            row = (index // 3) + 1
            showlegend = True if index == 0 else False

            # Real Plot
            fig.add_trace(
                go.Scatter(
                    y=item["real"],
                    mode="lines",
                    name="Real Segment",
                    line=dict(color="Lime"),
                    showlegend=showlegend,
                    fill="tozeroy",
                ),
                row=row,
                col=col,
            )
            # Simulator Plot
            fig.add_trace(
                go.Scatter(
                    y=item["simulate"],
                    mode="lines",
                    name="Simulated Segment",
                    line=dict(color="Red", width=4),
                    showlegend=showlegend,
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(
                title_text=name,
                row=row,
                col=col,
                titlefont=self.font,
                tickfont=dict(size=tickfont_size),
            )
            fig.update_yaxes(
                title_text="Memory (GB)",
                row=row,
                col=col,
                titlefont=self.font,
                tickfont=dict(size=tickfont_size),
            )

        default_width = image_size[0]
        default_height = image_size[1] * rows

        fig.update_layout(
            height=default_height,
            width=default_width,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font,
            ),
            showlegend=True,
            template=self._plotly_template,
            margin=dict(l=100, r=30, t=20, b=50, pad=6),
        )
        if view_mode is False:
            fig.write_image(
                file=self.get_image_dir(title),
                width=default_width,
                height=default_height,
            )
        return fig

    def plot_relative_error_in_box_diagram_with_verification_data(
        self,
        title: str,
        data_dir: Union[str, Path],
        optimiser: Optional[str] = None,
        yrange=(0, 250),
        image_size=(2000, 450),
        font_size=20,
        legend_font_size=18,
        legend_position=(0.95, 0.58),
        tickfont_size: int = 22,
        view_mode: bool = False,
        overall_median: bool = False,
        median_box_offset=(0.8, 0.92, 0.08),
    ):
        if view_mode:
            image_size = (1200, 350)
            font_size = 15
            legend_font_size = 12
            tickfont_size = 15

        self.font.update(dict(size=font_size))
        self.legend_font.update(dict(size=legend_font_size))
        df = self.data_processing(data_dir)
        if optimiser is not None:
            df = df[df["optimiser"] == optimiser]
        df["shortName"] = df["model"].apply(lambda x: x.split("/")[-1])
        model_order = sorted(df["shortName"].unique())
        df["error"] = df["error"] * 100

        # Filter out rows where is runtime = -1 and memory = -1, meaning that that estimator does not support this model
        df = df[~((df["runtime"] == -1) & (df["memory"] == -1))].copy()

        # Create a box plot grouped by 'Category'
        fig = px.box(
            df,
            x="shortName",
            y="error",
            color="tool",
            category_orders={"shortName": model_order},
            labels={"tool": "Estimator", "error": "Error", "model": "Model"},
            boxmode="group",
            color_discrete_map=self._color_scheme,
        )

        # Draw a triangle-up to exhibit that there is a outlier excess the 250%
        y_max = yrange[1]
        models_with_outliers = df[df["error"] > y_max]

        for _, row in models_with_outliers.iterrows():
            model = row["model"]
            name = row["tool"]
            x_offset_mapping = {
                ApproachedName.solution.value: -30,
                ApproachedName.DNNmem.value: -15,
                ApproachedName.SchedTune.value: 15,
                ApproachedName.LLmem.value: 30,
            }

            fig.add_annotation(
                x=model,
                y=y_max,
                text="▲",
                showarrow=False,
                font={**self.font, "color": self._color_scheme.get(name)},
                xanchor="center",
                yanchor="bottom",
                align="center",
                yshift=-20,
                xshift=x_offset_mapping.get(name, 0),
            )
        if overall_median:
            for i, name in enumerate(df["tool"].unique()):
                # Filter data for each 'name'
                name_filtered_df = df[df["tool"] == name]

                if not name_filtered_df.empty:
                    overall_median = name_filtered_df[
                        "error"
                    ].median()  # Calculate the overall median for this name

                    # Add annotation at the top-left of the plot, vertically stacked
                    fig.add_annotation(
                        x=median_box_offset[
                            0
                        ],  # Place near the left of the plot (use paper coordinates)
                        y=median_box_offset[1]
                        - (
                            i * median_box_offset[2]
                        ),  # Decrease y position for each annotation to vertically stack
                        text=f"{int(round(overall_median, 0))}%",
                        showarrow=False,
                        xref="paper",  # Use 'paper' coordinates for relative positioning
                        yref="paper",
                        font=dict(**self.font, color=self._color_scheme[name]),
                        xanchor="left",
                        yanchor="top",
                    )

        fig.update_traces(marker=dict(size=3))
        fig.update_xaxes(tickangle=15)

        default_width = image_size[0]
        default_height = image_size[1]
        fig.update_layout(
            font=self.font,
            xaxis_title=None,
            yaxis_title="Relative Error (%)",
            legend=dict(
                title=None,
                orientation="v",
                yanchor="bottom",
                y=legend_position[1],
                xanchor="center",
                x=legend_position[0],
                bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="LightGray",
                zeroline=False,
                titlefont=self.font,
                range=yrange,
                tickfont=dict(size=tickfont_size),
            ),
            xaxis=dict(
                showgrid=False, titlefont=self.font, tickfont=dict(size=tickfont_size)
            ),
            title_x=0.5,
            plot_bgcolor="white",
            width=default_width,  # Adjust width
            height=default_height,  # Adjust height
            template=self._plotly_template,
            margin=dict(l=60, r=20, t=20, b=50, pad=4),
        )
        if view_mode is False:
            # Show the plot
            fig.write_image(
                file=self.get_image_dir(title),
                width=default_width,
                height=default_height,
            )
        return fig

    def llm_plot_relative_error_in_box_diagram_with_verification_data(
        self,
        title: str,
        data_dir: Union[str, Path],
        optimiser: Optional[str] = None,
        yrange=(0, 250),
        image_size=(2000, 450),
        overall_median: bool = False,
        font_size=20,
        legend_font_size=18,
        tickfont_size: int = 22,
        view_mode: bool = False,
    ):
        if view_mode:
            image_size = (1200, 350)
            font_size = 15
            legend_font_size = 12
            tickfont_size = 15

        self.font.update(dict(size=font_size))
        self.legend_font.update(dict(size=legend_font_size))
        df = self.data_processing(data_dir)
        if optimiser is not None:
            df = df[df["optimiser"] == optimiser]
        model_order = sorted(df["model"].unique())
        df["error"] = df["error"] * 100

        # Create a box plot grouped by 'Category'
        fig = px.box(
            df,
            x="model",
            y="error",
            color="tool",
            category_orders={"model": model_order},
            labels={"tool": "Estimator", "error": "Error", "model": "Model"},
            boxmode="group",
            color_discrete_map=self._color_scheme,
        )

        # Draw a triangle-up to exhibit that there is a outlier excess the 250%
        y_max = yrange[1]
        models_with_outliers = df[df["error"] > y_max]

        # for _, row in models_with_outliers.iterrows():
        #     model = row["model"]
        #     name = row["tool"]
        #     x_offset_mapping = {
        #         "xMem (this paper)": -30,
        #         "DNNMem": -15,
        #         "SchedTune": 15,
        #         "LLMem": 30,
        #     }
        #
        #     fig.add_annotation(
        #         x=model,
        #         y=y_max,
        #         text="▲",
        #         showarrow=False,
        #         font={**self.font, "color": self._color_scheme.get(name)},
        #         xanchor="center",
        #         yanchor="bottom",
        #         align="center",
        #         yshift=-20,
        #         xshift=x_offset_mapping.get(name, 0),
        #     )
        if overall_median:
            y_offset = 0.92
            for i, name in enumerate(df["tool"].unique()):
                # Filter data for each 'name'
                name_filtered_df = df[df["tool"] == name]

                if not name_filtered_df.empty:
                    overall_median = name_filtered_df[
                        "error"
                    ].median()  # Calculate the overall median for this name

                    # Add annotation at the top-left of the plot, vertically stacked
                    fig.add_annotation(
                        x=0.82,  # Place near the left of the plot (use paper coordinates)
                        y=y_offset
                        - (
                            i * 0.08
                        ),  # Decrease y position for each annotation to vertically stack
                        text=f"{round(overall_median, 2)}%",
                        showarrow=False,
                        xref="paper",  # Use 'paper' coordinates for relative positioning
                        yref="paper",
                        font=dict(**self.font, color=self._color_scheme[name]),
                        xanchor="left",
                        yanchor="top",
                    )

        fig.update_traces(marker=dict(size=3))
        fig.update_xaxes(tickangle=15)

        default_width = image_size[0]
        default_height = image_size[1]
        fig.update_layout(
            font=self.font,
            xaxis_title=None,
            yaxis_title="Relative Error (%)",
            legend=dict(
                title=None,
                orientation="v",
                yanchor="bottom",
                y=0.58,
                xanchor="center",
                x=0.95,
                bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="LightGray",
                zeroline=False,
                titlefont=self.font,
                range=yrange,
                tickfont=dict(size=tickfont_size),
            ),
            xaxis=dict(
                showgrid=False, titlefont=self.font, tickfont=dict(size=tickfont_size)
            ),
            title_x=0.5,
            plot_bgcolor="white",
            width=default_width,  # Adjust width
            height=default_height,  # Adjust height
            template=self._plotly_template,
            margin=dict(l=60, r=20, t=20, b=50, pad=4),
        )
        if view_mode is False:
            # Show the plot
            fig.write_image(
                file=self.get_image_dir(title),
                width=default_width,
                height=default_height,
            )
        return fig

    def llm_plot_probability_estimation_vs_error_scatter_diagram_model_base(
        self,
        title: str,
        data_dir: Union[str, Path],
        optimiser: Optional[str] = None,
        group_by_list: tuple = ("model", "tool"),
        accurate_estimation_mode: bool = True,
        image_size=(500, 500),
        title_font_size=35,
        legend_font_size=22,
        tickfont_size=25,
        marker_size=25,
        quadrant_font_size=45,
        view_mode: bool = False,
    ):
        if view_mode:
            image_size = (600, 600)
            title_font_size = 20
            legend_font_size = 12
            tickfont_size = 20
            quadrant_font_size = 20
            marker_size = 12

        self.font.update(dict(size=title_font_size))
        self.legend_font.update(dict(size=legend_font_size))
        df = self.data_processing(data_dir)
        df["error"] = df["error"] * 100
        group_by_list = list(group_by_list)
        if optimiser is not None:
            df = df[df["optimiser"] == optimiser]
        # Step 1: Create oom_counts with OOM counts and probability
        result_field = "correct_estimation"
        if accurate_estimation_mode:
            result_field = "accurate_estimation"
        correctness_counts = (
            df.groupby(group_by_list)[result_field]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        if True not in correctness_counts.columns:
            correctness_counts[True] = 0
        if False not in correctness_counts.columns:
            correctness_counts[False] = 0

        correctness_counts.columns = group_by_list + [
            "Correct_Estimation_False_Count",
            "Correct_Estimation_True_Count",
        ]
        correctness_counts["probability"] = (
            correctness_counts["Correct_Estimation_False_Count"]
            / (
                correctness_counts["Correct_Estimation_False_Count"]
                + correctness_counts["Correct_Estimation_True_Count"]
            )
        ) * 100

        # Step 2: Aggregate 'error' by mean
        error_agg = df.groupby(group_by_list)["error"].median().reset_index()
        error_agg.rename(columns={"error": "Mean_Error"}, inplace=True)
        error_agg["Mean_Error"] = round(error_agg["Mean_Error"], 2)

        # Step 3: Merge the aggregated error data with oom_counts
        correctness_counts = correctness_counts.merge(
            error_agg, on=group_by_list, how="left"
        )
        fig = px.scatter(
            correctness_counts,
            x="probability",  # Probability of successful estimation on X-axis
            y="Mean_Error",  # Relative error on Y-axis
            labels={
                "probability": "Failed Estimation Probability  (%)",
                "Mean_Error": "Relative Error (%)",
                "name": "Estimator",
            },
            color="tool",  # Differentiates points by 'name' using color
            symbol="tool",  # Differentiates points by 'name' using marker symbols
            hover_data=[
                "Correct_Estimation_False_Count",
                "Correct_Estimation_True_Count",
                "Mean_Error",
            ],  # Additional info on hover
            color_discrete_map=self._color_scheme,
        )

        # Update all markers to have the same size
        fig.update_traces(marker=dict(size=marker_size))  # Set a fixed size, e.g., 12

        fig = self._add_four_quadrant(
            fig,
            (20, 20),
            100,
            font_size=quadrant_font_size,
        )

        # Customize the layout for better aesthetics
        default_width = image_size[0]
        default_height = image_size[1]
        fig.update_layout(
            font=self.font,
            legend=dict(
                title=None,
                orientation="v",
                yanchor="bottom",
                y=0.7,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font,
            ),
            xaxis=dict(
                title="Failed Estimation Probability (%)",
                titlefont=self.font,
                tickfont=dict(size=tickfont_size),
                range=(0, 100),
            ),
            yaxis=dict(
                title="Median of Relative Errors (%)",
                titlefont=self.font,
                tickfont=dict(size=tickfont_size),
                range=(0, 100),
            ),
            width=default_width,
            height=default_height,
            template=self._plotly_template,
            margin=dict(l=20, r=10, t=30, b=50),
        )
        if view_mode is False:
            # Show the plot
            fig.write_image(
                file=self.get_image_dir(title),
                width=default_width,
                height=default_height,
            )
        return fig

    def plot_probability_estimation_vs_error_scatter_diagram_model_base(
        self,
        title: str,
        data_dir: Union[str, Path],
        optimiser: Optional[str] = None,
        group_by_list: tuple = ("model", "tool"),
        accurate_estimation_mode: bool = True,
        image_size=(500, 500),
        title_font_size=35,
        legend_font_size=22,
        tickfont_size=25,
        marker_size=25,
        quadrant_font_size=45,
        view_mode: bool = False,
    ):
        if view_mode:
            image_size = (600, 600)
            title_font_size = 20
            legend_font_size = 12
            tickfont_size = 20
            quadrant_font_size = 20
            marker_size = 12

        self.font.update(dict(size=title_font_size))
        self.legend_font.update(dict(size=legend_font_size))
        df = self.data_processing(data_dir)
        df["error"] = df["error"] * 100
        # Filter out rows where is runtime = -1 and memory = -1, meaning that that estimator does not support this model
        df = df[~((df["runtime"] == -1) & (df["memory"] == -1))].copy()

        group_by_list = list(group_by_list)
        if optimiser is not None:
            df = df[df["optimiser"] == optimiser]
        # Step 1: Create oom_counts with OOM counts and probability
        result_field = "correct_estimation"
        if accurate_estimation_mode:
            result_field = "accurate_estimation"
        correctness_counts = (
            df.groupby(group_by_list)[result_field]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        if True not in correctness_counts.columns:
            correctness_counts[True] = 0
        if False not in correctness_counts.columns:
            correctness_counts[False] = 0

        correctness_counts.columns = group_by_list + [
            "Correct_Estimation_False_Count",
            "Correct_Estimation_True_Count",
        ]
        correctness_counts["probability"] = (
            correctness_counts["Correct_Estimation_False_Count"]
            / (
                correctness_counts["Correct_Estimation_False_Count"]
                + correctness_counts["Correct_Estimation_True_Count"]
            )
        ) * 100

        # Step 2: Aggregate 'error' by mean
        error_agg = (
            df[(df["real_oom"] == False) & (df["oom"] == False)]
            .groupby(group_by_list)["error"]
            .median()
            .reset_index()
        )
        error_agg.rename(columns={"error": "Mean_Error"}, inplace=True)
        error_agg["Mean_Error"] = round(error_agg["Mean_Error"], 2)

        # Step 3: Merge the aggregated error data with oom_counts
        correctness_counts = correctness_counts.merge(
            error_agg, on=group_by_list, how="left"
        )
        fig = px.scatter(
            correctness_counts,
            x="probability",  # Probability of successful estimation on X-axis
            y="Mean_Error",  # Relative error on Y-axis
            labels={
                "probability": "Failed Estimation Probability  (%)",
                "Mean_Error": "Relative Error (%)",
                "name": "Estimator",
            },
            color="tool",  # Differentiates points by 'name' using color
            symbol="tool",  # Differentiates points by 'name' using marker symbols
            hover_data=[
                "model",
                "Correct_Estimation_False_Count",
                "Correct_Estimation_True_Count",
                "Mean_Error",
            ],  # Additional info on hover
            color_discrete_map=self._color_scheme,
        )

        # Update all markers to have the same size
        fig.update_traces(marker=dict(size=marker_size))  # Set a fixed size, e.g., 12

        fig = self._add_four_quadrant(
            fig,
            (20, 20),
            correctness_counts["Mean_Error"].max(),
            font_size=quadrant_font_size,
        )

        # Customize the layout for better aesthetics
        default_width = image_size[0]
        default_height = image_size[1]
        fig.update_layout(
            font=self.font,
            legend=dict(
                title=None,
                orientation="v",
                yanchor="bottom",
                y=0.7,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font,
            ),
            xaxis=dict(
                title="Failed Estimation Probability (%)",
                titlefont=self.font,
                tickfont=dict(size=tickfont_size),
            ),
            yaxis=dict(
                title="Median of Relative Errors (%)",
                titlefont=self.font,
                tickfont=dict(size=tickfont_size),
                range=(0, max(correctness_counts["Mean_Error"])),
            ),
            width=default_width,
            height=default_height,
            template=self._plotly_template,
            margin=dict(l=20, r=10, t=30, b=50),
        )
        if view_mode is False:
            # Show the plot
            fig.write_image(
                file=self.get_image_dir(title),
                width=default_width,
                height=default_height,
            )
        return fig

    def plot_probability_estimation_vs_runtime_scatter_diagram_model_base(
        self,
        title: str,
        data_dir: Union[str, Path],
        optimiser: Optional[str] = None,
        group_by_list: list = ("model", "tool"),
        accurate_estimation_mode: bool = True,
        image_size=(500, 500),
        title_font_size=35,
        legend_font_size=22,
        tickfont_size=25,
        marker_size=25,
        view_mode: bool = False,
    ):
        if view_mode:
            image_size = (600, 600)
            title_font_size = 20
            legend_font_size = 12
            tickfont_size = 20
            quadrant_font_size = 20
            marker_size = 12
        self.font.update(dict(size=title_font_size))
        self.legend_font.update(dict(size=legend_font_size))
        df = self.data_processing(data_dir)
        group_by_list = list(group_by_list)
        if optimiser is not None:
            df = df[df["optimiser"] == optimiser]
        # Step 1: Create oom_counts with OOM counts and probability
        result_field = "correct_estimation"
        if accurate_estimation_mode:
            result_field = "accurate_estimation"

        # Step 1: Create oom_counts with OOM counts and probability
        correctness_counts = (
            df.groupby(group_by_list)[result_field]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        if True not in correctness_counts.columns:
            correctness_counts[True] = 0
        if False not in correctness_counts.columns:
            correctness_counts[False] = 0

        correctness_counts.columns = group_by_list + [
            "Correct_Estimation_False_Count",
            "Correct_Estimation_True_Count",
        ]
        correctness_counts["probability"] = (
            correctness_counts["Correct_Estimation_False_Count"]
            / (
                correctness_counts["Correct_Estimation_False_Count"]
                + correctness_counts["Correct_Estimation_True_Count"]
            )
        ) * 100

        # Step 2: Aggregate 'error' by mean
        error_agg = df.groupby(group_by_list)["runtime"].mean().reset_index()

        # Step 3: Merge the aggregated error data with oom_counts
        correctness_counts = correctness_counts.merge(
            error_agg, on=group_by_list, how="left"
        )
        fig = px.scatter(
            correctness_counts,
            x="probability",  # Probability of successful estimation on X-axis
            y="runtime",  # Relative error on Y-axis
            labels={
                "probability": "Failed Estimation Probability  (%)",
                "Mean_Error": "Runtime (s)",
                "name": "Estimator",
            },
            color="tool",  # Differentiates points by 'name' using color
            symbol="tool",  # Differentiates points by 'name' using marker symbols
            hover_data=[
                "Correct_Estimation_False_Count",
                "Correct_Estimation_True_Count",
                "runtime",
                "model",
            ],  # Additional info on hover
            color_discrete_map=self._color_scheme,
        )

        # Update all markers to have the same size
        fig.update_traces(marker=dict(size=marker_size))  # Set a fixed size, e.g., 12

        default_width = image_size[0]
        default_height = image_size[1]

        # Customize the layout for better aesthetics
        fig.update_layout(
            font=self.font,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font,
            ),
            xaxis=dict(
                title="Failed Estimation Probability (%)",
                titlefont=self.font,
                tickfont=dict(size=tickfont_size),
            ),
            yaxis=dict(
                title="Average Runtime (s)",
                titlefont=self.font,
                tickfont=dict(size=tickfont_size),
                type="log",
            ),
            width=default_width,
            height=default_height,
            template=self._plotly_template,
            margin=dict(l=30, r=10, t=30, b=30),
        )

        if view_mode is False:
            # Show the plot
            fig.write_image(
                file=self.get_image_dir(title),
                width=default_width,
                height=default_height,
            )
        return fig

    def plot_cdf_random_test_estimator_performance_score(
        self,
        title: str,
        data_dir: Union[str, Path],
        probability_weight: float = 0.7,
        error_weight: float = 0.3,
        image_size=(1800, 1200),
        font_size=50,
        legend_font_size=40,
        tickfont_size=40,
        line_width=10,
        view_mode: bool = False,
    ):
        if view_mode:
            image_size = (1000, 500)
            font_size = 25
            legend_font_size = 15
            line_width = 5
            tickfont_size = 20

        self.font.update(dict(size=font_size))
        self.legend_font.update(dict(size=legend_font_size))
        df = self.data_processing(data_dir)
        # Filter out rows where is runtime = -1 and memory = -1, meaning that that estimator does not support this model
        df = df[~((df["runtime"] == -1) & (df["memory"] == -1))].copy()

        # for 1st round of verification
        grouped = (
            df.groupby(["tool", "model", "optimiser"])
            .agg(
                failed_rate=(
                    "correct_estimation",
                    lambda x: (x == False).sum() / len(x),
                ),
                average_error=("error", "median"),
            )
            .reset_index()
        )

        fig = go.Figure()

        for tool_name, group in grouped.groupby("tool"):
            group["combined_score"] = (
                probability_weight * (group["failed_rate"])
                + error_weight * group["average_error"]
            )

            sorted_combined_score = np.sort(group["combined_score"].dropna())
            cdf = np.arange(1, len(sorted_combined_score) + 1) / len(
                sorted_combined_score
            )

            fig.add_trace(
                go.Scatter(
                    x=sorted_combined_score,
                    y=cdf,
                    mode="lines",
                    name=f"{tool_name}-1st",
                    line=dict(
                        color=self._color_scheme.get(tool_name),
                        dash="dashdot",
                        width=line_width,
                    ),
                )
            )

        # for 2nd round of verification
        grouped = (
            df.groupby(["tool", "model", "optimiser"])
            .agg(
                second_oom_failed_rate=(
                    "accurate_estimation",
                    lambda x: (x == False).sum() / len(x),
                ),
                average_second_error=("2nd_error", "median"),
            )
            .reset_index()
        )

        for tool_name, group in grouped.groupby("tool"):
            group["combined_score"] = (
                probability_weight * (group["second_oom_failed_rate"])
                + error_weight * group["average_second_error"]
            )

            sorted_combined_score = np.sort(group["combined_score"].dropna())
            cdf = np.arange(1, len(sorted_combined_score) + 1) / len(
                sorted_combined_score
            )

            fig.add_trace(
                go.Scatter(
                    x=sorted_combined_score,
                    y=cdf,
                    mode="lines",
                    name=f"{tool_name}-2nd",
                    line=dict(
                        color=self._color_scheme.get(tool_name), width=line_width
                    ),
                )
            )

        default_width = image_size[0]
        default_height = image_size[1]
        fig.update_layout(
            xaxis_title=f"Performance Score",
            yaxis_title="CDF",
            yaxis=dict(
                showgrid=True,
                gridcolor="LightGray",
                zeroline=False,
                titlefont=self.font,
                range=[0, 1],
                tickfont=dict(size=tickfont_size),
            ),
            xaxis=dict(
                showgrid=False,
                titlefont=self.font,
                range=[0, 1],
                tickfont=dict(size=tickfont_size),
            ),
            height=default_height,  # default height
            width=default_width,  # default width
            font=self.font,
            legend=dict(
                orientation="v",
                yanchor="bottom",
                y=0.1,
                xanchor="center",
                x=0.75,
                bgcolor="rgba(255,255,255,0.9)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font,
            ),
            margin=dict(l=50, r=30, t=40, b=50),
            showlegend=True,
            template=self._plotly_template,
        )
        if view_mode is False:
            fig.write_image(
                file=self.get_image_dir(title),
                width=default_width,
                height=default_height,
            )
        return fig

    def plot_random_test_memory_saved_bar_chart(
        self,
        title: str,
        data_dir: Union[str, Path],
        image_size=(800, 600),
        font_size=20,
        legend_font_size=18,
        legend_position=(0.95, 0.58),
        tickfont_size=25,
        overall: bool = False,
        view_mode: bool = False,
    ):
        if view_mode:
            image_size = (1000, 400)
            font_size = 20
            legend_font_size = 15
            tickfont_size = 15

        if overall:
            group_by = ["tool"]
        else:
            group_by = ("model", "tool")
        self.font.update(dict(size=font_size))
        self.legend_font.update(dict(size=legend_font_size))
        df = self.data_processing(data_dir)
        df = df[(df["runtime"] != -1) & (df["memory"] != -1)]

        group_by = list(group_by)
        total_df = (
            df.groupby(group_by)
            .agg(
                total_runs=("memory", "sum"),
            )
            .reset_index()
        )

        memory_save = (
            df[(df["accurate_estimation"] == True)]
            .groupby(group_by)
            .agg(
                save_memory_sum=("save_memory", "sum"),
                success_count=("save_memory", "count"),
            )
            .reset_index()
        )

        # Calculate memory_waste
        memory_waste = (
            df[(df["accurate_estimation"] == False)]
            .groupby(group_by)
            .agg(
                total_assign_memory=("memory", "sum"),
                failed_count=("memory", "count"),
            )
            .reset_index()
        )

        # Merge memory_save and memory_waste
        merged_memory = total_df.merge(memory_save, on=group_by, how="left")
        merged_memory = merged_memory.merge(memory_waste, on=group_by, how="left")

        # Fill NaN values with 0
        merged_memory = merged_memory.fillna(0)

        # Calculate average
        merged_memory["average"] = (
            merged_memory["save_memory_sum"] - merged_memory["total_assign_memory"]
        ) / (merged_memory["success_count"] + merged_memory["failed_count"])

        merged_memory["average"] = merged_memory["average"] / 1024**3

        # Sort values
        merged_memory = merged_memory.sort_values(by="save_memory_sum", ascending=False)
        model_order = sorted(df["model"].unique())

        if overall:
            # Plot bar chart
            fig = px.bar(
                merged_memory,
                x="tool",
                y="average",
                color="tool",
                barmode="group",
                color_discrete_map=self._color_scheme,
                category_orders={"model": model_order},
            )
        else:
            # Plot bar chart
            fig = px.bar(
                merged_memory,
                x="model",
                y="average",
                color="tool",
                barmode="group",
                color_discrete_map=self._color_scheme,
                category_orders={"model": model_order},
            )

        default_width = image_size[0]
        default_height = image_size[1]
        fig.update_xaxes(tickangle=30)
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Avg. Saved Memory (GB)",
            yaxis=dict(
                showgrid=True,
                gridcolor="LightGray",
                zeroline=False,
                titlefont=self.font,
                tickfont=dict(size=tickfont_size),
            ),
            xaxis=dict(
                showgrid=False, titlefont=self.font, tickfont=dict(size=tickfont_size)
            ),
            height=default_height,  # default height
            width=default_width,  # default width
            font=self.font,
            legend=dict(
                title=None,
                orientation="v",
                yanchor="bottom",
                y=legend_position[1],
                xanchor="center",
                x=legend_position[0],
                bgcolor="rgba(255,255,255,0.7)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font,
                itemsizing="trace",
            ),
            showlegend=True,
            template=self._plotly_template,
            margin=dict(
                l=50,
                r=20,
                t=20,
                b=50,
            ),
        )
        if view_mode is False:
            fig.write_image(
                file=self.get_image_dir(title),
                width=default_width,
                height=default_height,
            )
        return fig, merged_memory

    def display_summary(self, data_dir: Union[str, Path], with_llmem: bool = False):
        df = self.summarize_data(data_dir, with_llmem)
        print()

    def summarize_data(
        self,
        data_dir: Union[str, Path],
    ):
        df = self.data_processing(data_dir)
        # Calculating median error by 'tool'
        groupby_list = ["tool"]

        # Relative Error
        median_error = (
            df[(df["real_oom"] == False) & (df["oom"] == False)]
            .groupby(groupby_list)["error"]
            .median()
            .reset_index()
        )
        median_error.columns = groupby_list + ["Median Error"]
        median_error["Median Error"] = median_error["Median Error"] * 100

        # Probability of Estimation Failure
        correctness_counts = (
            df.groupby(groupby_list)["accurate_estimation"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        if True not in correctness_counts.columns:
            correctness_counts[True] = 0
        if False not in correctness_counts.columns:
            correctness_counts[False] = 0

        correctness_counts.columns = groupby_list + [
            "Correct_Estimation_False_Count",
            "Correct_Estimation_True_Count",
        ]
        correctness_counts["probability"] = (
            correctness_counts["Correct_Estimation_False_Count"]
            / (
                correctness_counts["Correct_Estimation_False_Count"]
                + correctness_counts["Correct_Estimation_True_Count"]
            )
        ) * 100

        # Runtime
        runtime = df.groupby(groupby_list)["runtime"].mean().reset_index()
        runtime["runtime"] = runtime["runtime"] / 1000**3

        # GPU Memory conservation
        total_df = (
            df.groupby(groupby_list)
            .agg(
                total_runs=("memory", "sum"),
            )
            .reset_index()
        )
        memory_save = (
            df[(df["accurate_estimation"] == True)]
            .groupby(groupby_list)
            .agg(
                save_memory_sum=("save_memory", "sum"),
                success_count=("save_memory", "count"),
            )
            .reset_index()
        )

        # Calculate memory_waste
        memory_waste = (
            df[(df["accurate_estimation"] == False)]
            .groupby(groupby_list)
            .agg(
                total_assign_memory=("memory", "sum"),
                failed_count=("memory", "count"),
            )
            .reset_index()
        )

        # Merge memory_save and memory_waste
        merged_memory = total_df.merge(memory_waste, on=groupby_list, how="left")
        merged_memory = merged_memory.merge(memory_save, on=groupby_list, how="left")

        # Fill NaN values with 0
        merged_memory = merged_memory.fillna(0)

        # Calculate memory conserved average
        merged_memory["GPU Memory"] = (
            merged_memory["save_memory_sum"] - merged_memory["total_assign_memory"]
        ) / (merged_memory["success_count"] + merged_memory["failed_count"])

        merged_memory["GPU Memory"] = merged_memory["GPU Memory"] / 1024**3

        # 1st validation performance CDF
        df_dict = {"tool": [], "performance_score_1": []}
        first_perf_score = (
            df.groupby(["tool", "model", "optimiser"])
            .agg(
                failed_rate=(
                    "correct_estimation",
                    lambda x: (x == False).sum() / len(x),
                ),
                average_error=("error", "median"),
            )
            .reset_index()
        )

        for tool_name, group in first_perf_score.groupby("tool"):
            group["combined_score"] = (
                0.7 * (group["failed_rate"]) + 0.3 * group["average_error"]
            )

            sorted_combined_score = np.sort(group["combined_score"].dropna())
            cdf = np.arange(1, len(sorted_combined_score) + 1) / len(
                sorted_combined_score
            )

            target_cdf = 0.95
            closest_index = np.abs(cdf - target_cdf).argmin()
            closest_score = sorted_combined_score[closest_index]
            df_dict["tool"].append(tool_name)
            df_dict["performance_score_1"].append(closest_score)
        first_ps_df = pd.DataFrame(df_dict)

        # 2nd validation performance CDF
        second_perf_score = (
            df.groupby(["tool", "model", "optimiser"])
            .agg(
                second_oom_success_rate=(
                    "accurate_estimation",
                    lambda x: (x == True).sum() / len(x),
                ),
                average_second_error=("2nd_error", "median"),
            )
            .reset_index()
        )
        df_dict = {"tool": [], "performance_score_2": []}
        for tool_name, group in second_perf_score.groupby("tool"):
            group["combined_score"] = (
                0.7 * (1 - group["second_oom_success_rate"])
                + 0.3 * group["average_second_error"]
            )

            sorted_combined_score = np.sort(group["combined_score"].dropna())
            cdf = np.arange(1, len(sorted_combined_score) + 1) / len(
                sorted_combined_score
            )

            target_cdf = 0.95
            closest_index = np.abs(cdf - target_cdf).argmin()
            closest_score = sorted_combined_score[closest_index]
            df_dict["tool"].append(tool_name)
            df_dict["performance_score_2"].append(closest_score)
        second_ps_df = pd.DataFrame(df_dict)

        # Displaying the result
        merged_df = correctness_counts.merge(median_error, on=groupby_list, how="left")
        merged_df = merged_df.merge(merged_memory, on=groupby_list, how="left")
        merged_df = merged_df.merge(runtime, on=groupby_list, how="left")
        merged_df = merged_df.merge(first_ps_df, on=groupby_list, how="left")
        merged_df = merged_df.merge(second_ps_df, on=groupby_list, how="left")

        # Calculate the average of the average memory saved
        summarized_result = {
            "key": ["xMem", "Baselines Average", "Improvement (%)"],
        }
        field_list = [
            "Median Error",
            "probability",
            "GPU Memory",
            # "performance_score_1",
            # "performance_score_2",
            "runtime",
        ]
        field_map = {
            "probability": "probability (%)",
            "GPU Memory": "GPU Memory (GB)",
            "Median Error": "Median Error (%)",
            "runtime": "Average Runtime (s)",
        }
        for field in field_list:
            dnnmem_value = merged_df[merged_df["tool"] == ApproachedName.DNNmem.value][
                field
            ].values[0]
            schedtune_value = merged_df[
                merged_df["tool"] == ApproachedName.SchedTune.value
            ][field].values[0]
            llmem_value = merged_df[merged_df["tool"] == ApproachedName.LLmem.value][
                field
            ].values[0]
            xmem_value = merged_df[merged_df["tool"] == ApproachedName.solution.value][
                field
            ].values[0]

            # average
            sum_list = [dnnmem_value, schedtune_value, llmem_value]
            average_value = sum(sum_list) / len(sum_list)
            improved_value = (xmem_value - average_value) / average_value * 100
            # Except for GPU memory conservation, the lower number is better. So, we need to flip the sign.
            if field != "GPU Memory":
                improved_value = -improved_value

            summarized_result[field] = [
                round(xmem_value, 2),
                round(average_value, 2),
                round(improved_value, 2),
            ]

        tool_summary = merged_df[merged_df[["tool"] + field_list].columns].round(2)
        # tool_summary = tool_summary.sort_values(
        #     by="performance_score_1", ascending=False
        # )
        summarized_result = pd.DataFrame(summarized_result)

        # standardize the columns name
        tool_summary.columns = tool_summary.columns.map(
            lambda col: field_map.get(col, col)
        )
        summarized_result.columns = summarized_result.columns.map(
            lambda col: field_map.get(col, col)
        )

        return tool_summary, summarized_result
