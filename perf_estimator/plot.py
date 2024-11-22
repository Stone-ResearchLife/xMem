import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import copy
import numpy as np
from networkx.algorithms.bipartite.basic import color
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Union, List, Dict, Optional
from perf_estimator.utilis import filter_files


class ExperimentPlot:
    def __init__(self):
        self._color_scheme = {
            'xMem (this paper)': '#636EFA',
            'DNNMem': '#EF553B',
            'SchedTune': '#00CC96',
            'LLMem': '#AB63FA'
        }
        self._name_map = {
            'solution': 'xMem (this paper)',
            'dnnmem': 'DNNMem',
            'schedtune': 'SchedTune',
            'llmem': 'LLMem'
        }
        self.font = dict(
            family = "Times New Roman",
            size=18
        )
        self.legend_font = dict(
            family = "Times New Roman",
            size=12
        )
        self._plotly_template = 'plotly_white'
        self._cache: Dict[str, pd.DataFrame] = {}
        self._work_dir = Path().home().joinpath("Documents/003-Thesis/CrossMemoryEstimator")
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._output_format = "pdf"

    def get_image_dir(self, title: str) -> Path:
        file_name = f"{title}.{self._output_format}"
        _dir = self._work_dir.joinpath("images")
        if _dir.exists() is False:
            _dir.mkdir(parents=True, exist_ok=True)
        return _dir.joinpath(file_name)

    def _evaluation_data2dict(self, data_file: Union[str, Path]) -> List[dict]:
        data_file = data_file if isinstance(data_file, Path) else Path(data_file)
        with open(data_file, 'r') as f:
            data = json.load(f)
        # sample of dir name: recurrence-ConvNeXt-30-1 -> recurrence-[model]-[batch_size]-[device id]
        split_name = str(data_file.parent.parent.parent.name).split("-")
        if len(split_name) == 4:
            model = split_name[1]
            optimiser = "N/A"
            batch = split_name[2]
            device = split_name[3]
        elif len(split_name) > 4:
            model = split_name[1]
            optimiser = split_name[2]
            batch = split_name[3]
            device = split_name[4]
        else:
            model = data['train info']['model']
            optimiser = data['train info'].get('optimizer', "N/A")
            batch = data['train info']['batch']
            device = data['train info']['device']

        zero_grad_mode = data['train info'].get('zero_grad_mode', 1)

        del data['config'], data['train info']
        output = []
        for key, value in data.items():
            _org_data = {
                "model": model,
                "batch": batch,
                "device": device,
                "name": key,
                "optimizer": optimiser,
                "zero_grad_mode": zero_grad_mode,
                "runtime": value['runtime'],
                "estimated": value['memory'] / 1024 ** 3,
                "ground": value['nvml_ground'] / 1024 ** 3,
                "error": value['error-nvml'],
                "v_error": value.get("error-nvml-verification", None),
                "OOM": value["OOM"]
            }
            output.append(_org_data)
        return output

    def dir2pd(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        assert data_dir.is_dir() is True
        if str(data_dir) not in self._cache.keys():
            print(f"Processing {data_dir}...")
            # generate evaluation df since it is not in cache
            evaluation_files = filter_files("evaluation_result.json", str(data_dir), fuzz=False)
            pd_list = []
            found_file_num = len(evaluation_files)
            print(f"Found {found_file_num} evaluation files")
            for index,eva in enumerate(evaluation_files):
                print(f"\rProcessing {index}/{len(evaluation_files)}", end="")
                output_list = self._evaluation_data2dict(eva)
                pd_list.extend(output_list)

            df = pd.DataFrame(pd_list)
            # replace name into pre-set name
            df['name'] = df['name'].replace(self._name_map)
            df['error'] = round(df['error'] * 100, 2)
            df['color'] = df['name'].replace(self._color_scheme)
            self._cache[str(data_dir)] = df
        else:
            print(f"Cache hit for {data_dir}")
        return self._cache[str(data_dir)].copy()

    def plot_estimation_error_in_box_diagram(
            self,
            title: str,
            data_dir: Union[str, Path],
            optimiser: Optional[str] = None,
            yrange = (0, 250),
            image_size = (1000, 500)
    ):
        df = self.dir2pd(data_dir)
        if optimiser is not None:
            df = df[df['optimizer'] == optimiser]
        model_order = sorted(df['model'].unique())

        # Create a box plot grouped by 'Category'
        fig = px.box(
            df,
            x='model',
            y='error',
            color='name',
            category_orders={'model': model_order},
            labels={'name': 'Name', 'error': 'Error', 'model': 'Model'},
            boxmode='group',
            color_discrete_map=self._color_scheme
        )

        # # Iterate over each model to add an annotation
        # target_estimator_name = "xMem (this paper)"
        # for model in model_order:
        #     names = df['name'].unique()
        #     for i, name in enumerate(names):
        #         if name != target_estimator_name:  # Skip the current model
        #             continue
        #         # Filter the data for the specific model and name
        #         filtered_df = df[(df['model'] == model) & (df['name'] == name)]
        #
        #         if not filtered_df.empty:
        #             median_value = filtered_df['error'].median()  # Get the median value for annotation
        #
        #             # Add annotation
        #             fig.add_annotation(
        #                 x=model,
        #                 y=10,  # Place the text below the x-axis
        #                 text=f"{int(median_value)}%",
        #                 showarrow=False,
        #                 xanchor='center',
        #                 yanchor='top',
        #                 font=dict(size=10, color=self._color_scheme[name]),
        #                 yshift=-15 * (i + 1)  # Shift downward to avoid overlapping with other labels
        #             )

        # Draw a triangle-up to exhibit that there is a outlier excess the 250%
        # y_max = yrange[1]
        # models_with_outliers = df[df['error'] > y_max]['model'].unique()
        #
        # for model in models_with_outliers:
        #     fig.add_annotation(
        #         x=model,
        #         y=y_max,
        #         text="▲",
        #         showarrow=False,
        #         font={**self.font, "color": 'red'},
        #         xanchor="center",
        #         yanchor="bottom",
        #         align="center",
        #         yshift=-20
        #     )
        #
        # arrow_trace = go.Scatter(
        #     x=[None],
        #     y=[None],
        #     mode='markers+text',
        #     marker=dict(
        #         symbol='triangle-up',  # 用向上三角形表示箭头
        #         size=10,
        #         color='red'
        #     ),
        #     text=[f"Outliers > {y_max}"],  # 添加文本
        #     # textposition="right",
        #     name=f'Outliers > {y_max}'  # 图例中的名称
        # )

        y_offset = 1
        for i, name in enumerate(df['name'].unique()):
            # Filter data for each 'name'
            name_filtered_df = df[df['name'] == name]

            if not name_filtered_df.empty:
                overall_median = name_filtered_df['error'].median()  # Calculate the overall median for this name

                # Add annotation at the top-left of the plot, vertically stacked
                fig.add_annotation(
                    x=0.01,  # Place near the left of the plot (use paper coordinates)
                    y=y_offset - (i * 0.05),  # Decrease y position for each annotation to vertically stack
                    text=f"{int(overall_median)}%: {name}",
                    showarrow=False,
                    xref='paper',  # Use 'paper' coordinates for relative positioning
                    yref='paper',
                    font=dict(size=12, color=self._color_scheme[name]),
                    xanchor='left',
                    yanchor='top'
                )

        fig.update_traces(marker=dict(size=3))

        default_width = image_size[0]
        default_height = image_size[1]
        fig.update_layout(
            font=self.font,
            xaxis_title="Models",
            yaxis_title="Relative Error (%)",
            legend=dict(
                title={
                    'text': "Estimator",
                    'font': {**self.legend_font, 'weight': "bold"}
                },
                orientation="h",
                yanchor="bottom",
                y=0.9,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font
            ),
            yaxis=dict(showgrid=True, gridcolor='LightGray', zeroline=False, titlefont=self.font, range=yrange),
            xaxis=dict(showgrid=False, titlefont=self.font),
            title_x=0.5,
            plot_bgcolor='white',
            width=default_width,  # Adjust width
            height=default_height,  # Adjust height
            template=self._plotly_template
        )
        # Show the plot
        fig.write_image(
            file=self.get_image_dir(title),
            width=default_width,
            height=default_height
        )
        return fig

    def plot_optimiser_view_error_in_box_diagram(
            self,
            title: str,
            data_dir: Union[str, Path],
            optimiser: Optional[str] = None,
            only_count_non_oom: bool = False
    ):
        df = self.dir2pd(data_dir)
        if optimiser is not None:
            df = df[df['optimizer'] == optimiser]
        if only_count_non_oom:
            df = df[df['OOM'] == False]
        model_order = df.groupby('error')['error'].median().sort_values().index

        # Create a box plot grouped by 'Category'
        fig = px.box(
            df,
            x='optimizer',
            y='error',
            color='name',
            category_orders={'model': model_order},
            labels={'name': 'Name', 'error': 'Error', 'optimizer': 'Optimiser'},
            boxmode='group',
            color_discrete_map=self._color_scheme
        )

        fig.update_traces(marker=dict(size=3))

        default_width = 1000
        default_height = 500
        fig.update_layout(
            font=self.font,
            xaxis_title="Models",
            yaxis_title="Relative Error (%)",
            legend=dict(
                title={
                    'text': "Estimator",
                    'font': {**self.legend_font, 'weight': "bold"}
                },
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font
            ),
            yaxis=dict(showgrid=True, gridcolor='LightGray', zeroline=False, titlefont=self.font),
            xaxis=dict(showgrid=False, titlefont=self.font),
            title_x=0.5,
            plot_bgcolor='white',
            width=default_width,  # Adjust width
            height=default_height,  # Adjust height
            template=self._plotly_template
        )
        # Show the plot
        fig.write_image(
            file=self.get_image_dir(title),
            width=default_width,
            height=default_height
        )
        return fig

    def plot_probability_estimation_vs_error_scatter_diagram_model_base(
            self,
            title: str,
            data_dir: Union[str, Path],
            optimiser: Optional[str] = None
    ):
        df = self.dir2pd(data_dir)
        if optimiser is not None:
            df = df[df['optimizer'] == optimiser]
        # Step 1: Create oom_counts with OOM counts and probability
        oom_counts = df.groupby(['model', 'name'])['OOM'].value_counts().reset_index()
        if True not in oom_counts.columns:
            oom_counts[True] = 0
        oom_counts.columns = ['model', 'name', 'OOM_False_Count', 'OOM_True_Count']
        oom_counts['probability'] = (oom_counts['OOM_True_Count'] / (
                    oom_counts['OOM_False_Count'] + oom_counts['OOM_True_Count'])) * 100

        # Step 2: Aggregate 'error' by mean
        error_agg = df.groupby(['model', 'name'])['error'].mean().reset_index()
        error_agg.rename(columns={'error': 'Mean_Error'}, inplace=True)
        error_agg['Mean_Error'] = round(error_agg['Mean_Error'], 2)

        # Step 3: Merge the aggregated error data with oom_counts
        oom_counts = oom_counts.merge(error_agg, on=['model', 'name'], how='left')
        fig = px.scatter(
            oom_counts,
            x='probability',  # Probability of successful estimation on X-axis
            y='Mean_Error',  # Relative error on Y-axis
            labels={
                'probability': 'OOM Probability  (%)',
                'Mean_Error': 'Relative Error (%)',
                'name': 'Estimator'
            },
            color='name',  # Differentiates points by 'name' using color
            symbol='name',  # Differentiates points by 'name' using marker symbols
            hover_data=['OOM_False_Count', 'OOM_True_Count', 'Mean_Error', 'model'],  # Additional info on hover
            color_discrete_map=self._color_scheme
        )

        # Update all markers to have the same size
        fig.update_traces(marker=dict(size=12))  # Set a fixed size, e.g., 12

        # Customize the layout for better aesthetics
        default_width = 500
        default_height = 500
        fig.update_layout(
            font=self.font,
            legend=dict(
                title={
                    'text': "Estimator",
                    'font': {**self.legend_font, 'weight': "bold"}
                },
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font
            ),
            xaxis=dict(title='OOM Probability (%)', titlefont=self.font),
            yaxis=dict(title='Relative Error (%)', titlefont=self.font),
            width=default_width,
            height=default_height,
            template=self._plotly_template,
        )

        # Show the plot
        fig.write_image(
            file=self.get_image_dir(title),
            width=default_width,
            height=default_height
        )
        return fig


    def plot_probability_estimation_vs_runtime_scatter_diagram_model_base(
            self,
            title: str,
            data_dir: Union[str, Path],
            optimiser: Optional[str] = None
    ):
        df = self.dir2pd(data_dir)
        if optimiser is not None:
            df = df[df['optimizer'] == optimiser]
        # Step 1: Create oom_counts with OOM counts and probability
        oom_counts = df.groupby(['model', 'name'])['OOM'].value_counts().reset_index()
        if True not in oom_counts.columns:
            oom_counts[True] = 0
        oom_counts.columns = ['model', 'name', 'OOM_False_Count', 'OOM_True_Count']
        oom_counts['probability'] = (oom_counts['OOM_True_Count'] / (
                oom_counts['OOM_False_Count'] + oom_counts['OOM_True_Count'])) * 100

        # Step 2: Aggregate 'error' by mean
        error_agg = df.groupby(['model', 'name'])['runtime'].mean().reset_index()
        error_agg.rename(columns={'runtime': 'Runtime'}, inplace=True)

        # Step 3: Merge the aggregated error data with oom_counts
        oom_counts = oom_counts.merge(error_agg, on=['model', 'name'], how='left')
        fig = px.scatter(
            oom_counts,
            x='probability',  # Probability of successful estimation on X-axis
            y='Runtime',  # Relative error on Y-axis
            labels={
                'probability': 'OOM Probability  (%)',
                'Mean_Error': 'Runtime (s)',
                'name': 'Estimator'
            },
            color='name',  # Differentiates points by 'name' using color
            symbol='name',  # Differentiates points by 'name' using marker symbols
            hover_data=['OOM_False_Count', 'OOM_True_Count', 'Runtime', 'model'],  # Additional info on hover
            color_discrete_map=self._color_scheme
        )

        # Update all markers to have the same size
        fig.update_traces(marker=dict(size=12))  # Set a fixed size, e.g., 12

        default_width = 500
        default_height = 500

        # Customize the layout for better aesthetics
        fig.update_layout(
            font=self.font,
            legend=dict(
                title={
                    'text': "Estimator",
                    'font': {**self.legend_font, 'weight': "bold"}
                },
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font
            ),
            xaxis=dict(title='OOM Probability (%)', titlefont=self.font),
            yaxis=dict(
                title='Runtime (s)',
                titlefont=self.font,
                type="log",
                range=[-1, 3]
            ),
            width=default_width,
            height=default_height,
            template=self._plotly_template,
        )

        # Show the plot
        fig.write_image(
            file=self.get_image_dir(title),
            width=default_width,
            height=default_height
        )
        return fig

    def plot_probability_estimation_vs_error_scatter_diagram_run_base(
            self,
            title: str,
            data_dir: Union[str, Path],
            group_by: List[str] = None,
    ):
        df = self.dir2pd(data_dir)
        if group_by is not None:
            df = df.groupby(group_by).mean().reset_index()
        fig = px.scatter(
            df,
            x='error',
            y='runtime',
            labels={'error': 'Error', 'runtime': 'Runtime'},
            symbol='name',
            color='name',
            symbol_sequence=['star', 'star-triangle-up', 'diamond', 'cross'],
            color_discrete_map=self._color_scheme
        )
        fig.update_traces(
            marker=dict(size=12, opacity=0.5),  # 所有图形默认透明度
        )
        fig.update_traces(
            selector=dict(symbol='star-open'),
            marker=dict(size=15, opacity=1, color='blue')  # 星星图形的颜色和大小
        )

        fig.update_layout(
            legend=dict(
                title={
                    'text': "Estimator",
                    'font': self.font
                },
                x=0.8,  # Position legend x-coord
                y=1,  # Position legend y-coord
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.font
            ),
            width=800,
            height=600,
            legend_title_text='Name',
            xaxis=dict(title='Error (%)'),
            yaxis=dict(title='Runtime (s)', type="log"),
            template=self._plotly_template
        )
        fig.write_image(
            file=self.get_image_dir(title),
            width=800,
            height=600
        )
        return fig

    def plot_probability_figures_in_one(
            self,
            title: str,
            data_dir: Union[str, Path],
            optimiser: Optional[str] = None
    ):
        # Assume fig1 and fig2 are created already
        fig1 = self.plot_probability_estimation_vs_error_scatter_diagram_model_base(f"{title}-1", data_dir, optimiser)
        fig2 = self.plot_probability_estimation_vs_runtime_scatter_diagram_model_base(f"{title}-2", data_dir, optimiser)

        # Create the subplot layout
        fig = make_subplots(rows=1, cols=2)

        # Add traces from fig1 to the first column
        for trace in fig1.data:
            trace.marker.size = 10
            fig.add_trace(trace, row=1, col=1)

        # Add traces from fig2 to the second column, hiding legend for fig2
        for trace in fig2.data:
            trace.marker.size = 10
            fig.add_trace(trace.update(showlegend=False), row=1, col=2)

        # Update axis properties for fig1 (Success Rate vs Relative Error)
        fig.update_xaxes(title_text="OOM Probability (%)", row=1, col=1, titlefont=self.font)
        fig.update_yaxes(title_text="Relative Error (%)", range=[0, 100], row=1, col=1, titlefont={**self.font, "weight": "bold"})

        # Update axis properties for fig2 (Success Rate vs Runtime)
        fig.update_xaxes(title_text="OOM Probability (%)", row=1, col=2, titlefont=self.font)
        fig.update_yaxes(title_text="Runtime (s)", type="log", row=1, col=2, tickvals=[10, 100, 1000], titlefont={**self.font, "weight": "bold"})

        # Set layout properties for the combined figure
        default_width = 1000
        default_height = 500
        fig.update_layout(
            height=default_height,
            width=default_width,
            legend=dict(
                title={
                    'text': "Estimator",
                    'font': {'weight': "bold", **self.legend_font}
                },
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font
            ),
            showlegend=True,
            template=self._plotly_template
        )

        # Show the combined figure
        fig.write_image(
            file=self.get_image_dir(title),
            width=default_width,
            height=default_height
        )
        return fig



    def plot_fluctuation_of_result_box(self, title: str, data_dir: Union[str, Path]):
        df = self.dir2pd(data_dir)
        # Create a box plot grouped by 'Category'
        fig = px.box(df, x='name', y='error', color='model',
                     title='Fluctuation of Estimated Results in different models',
                     labels={'name': 'Name', 'error': 'Error', "model": "Model"},
                     boxmode='group',
                     color_discrete_map=self._color_scheme
                     )

        fig.update_layout(
            title_font_size=18,
            font=dict(size=14),
            xaxis_title="Estimator",
            yaxis_title="Error(%)",
            legend_title="Model",
            yaxis=dict(showgrid=True, gridcolor='LightGray', zeroline=False),
            xaxis=dict(showgrid=False),
            title_x=0.5,
            plot_bgcolor='white',
            width=1000,  # Adjust width
            height=600,  # Adjust height,
            template=self._plotly_template
        )
        # Show the plot
        fig.show()

    def plot_memory_fluctuation_by_changing_zero_out(self, title, data_dir: Union[str, Path]):
        from perf_estimator.snapshot import SnapshotAnalyser
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
                "trace": [trace / 1024 ** 3 for trace in traces],
                "segment": [seg / 1024 ** 3 for seg in segs]
            }

        num_plots = len(outputs.keys())
        cols = min(num_plots, 3)
        rows = 1 if num_plots <= 3 else ((num_plots - 1) // 3) + 1

        fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, vertical_spacing=0.1)

        for index, (name, item) in enumerate(outputs.items()):
            col = (index % 3) + 1
            row = (index // 3) + 1
            beforeBW_data = item['beforeBW']
            normal_data = item['normal']
            showlegend = True if index == 0 else False

            # Point 0
            fig.add_trace(
                go.Scatter(
                    y=normal_data['segment'],
                    mode='lines',
                    name='Segment (point0)',
                    line=dict(color='Red'),
                    showlegend=showlegend,
                ),
                row=row,
                col=col
            )
            # Point 1
            fig.add_trace(
                go.Scatter(
                    y=beforeBW_data['segment'],
                    mode='lines',
                    name='Segment (point1)',
                    line=dict(color='Lime', dash="dash"),
                    showlegend=showlegend,
                ),
                row=row,
                col=col
            )

            # trace - Point 1
            fig.add_trace(
                go.Scatter(
                    y=beforeBW_data['trace'],
                    mode='lines',
                    name='Tensor (point1)',
                    line=dict(color='Lime'),
                    showlegend=showlegend,
                    fill="tozeroy"
            ),
                row=row,
                col=col
            )
            # trace - Point 0
            fig.add_trace(
                go.Scatter(
                    y=normal_data['trace'],
                    mode='lines',
                    name='Tensor (point0)',
                    line=dict(color='Red'),
                    showlegend = showlegend,
                    fill="tozeroy"
                ),
                row=row,
                col=col
            )

            fig.update_xaxes(title_text=name, row=row, col=col)
            fig.update_yaxes(title_text="Memory (GB)", row=row, col=col)


        default_height = 400 * rows
        default_width = 800
        fig.update_layout(
            height=default_height,
            width=default_width,
            legend=dict(
                title={
                    'text': "Memory Type",
                    'font': {'weight': "bold", **self.legend_font}
                },
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font
            ),
            showlegend=True,
            template=self._plotly_template,
        )
        fig.write_image(
            file=self.get_image_dir(title),
            width=default_width,
            height=default_height
        )
        return fig

    def plot_assessment_result_between_snapshot_and_simulation(
            self,
            title: str,
            data_dir: Union[str, Path]
    ):
        if str(data_dir) not in self._cache.keys():
            from perf_estimator.allocator import AllocatorSim
            from perf_estimator.snapshot import SnapshotAnalyser
            max_gpu_memory_in_gb = 7.6
            max_sample = 23000
            _sim = AllocatorSim(max_allocated_memory_gb=max_gpu_memory_in_gb)
            pickles = filter_files(".pickle", data_dir, fuzz=True)
            outputs = {}
            for index, p in enumerate(pickles):
                p = Path(p)
                model_name = p.name.split(".")[0]
                snapshot_analyser = SnapshotAnalyser(str(p))
                _result_allocate = _sim.simulate(snapshot_analyser, segment_plot=False)
                memory_data = _sim.compare_segment_allocated_memory(snapshot_analyser, _result_allocate)
                # Convert to GB
                memory_data["simulate"] = [mem / 1024 ** 3 for mem in memory_data["simulate"][0:max_sample]]
                memory_data["real"] = [mem / 1024 ** 3 for mem in memory_data["real"][0:max_sample]]
                outputs[model_name] = memory_data
            self._cache[str(data_dir)] = outputs

        outputs = self._cache[str(data_dir)]
        num_plots = len(outputs.keys())
        cols = min(num_plots, 3)
        rows = 1 if num_plots <= 3 else ((num_plots - 1) // 3) + 1

        fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, vertical_spacing=0.1)

        for index, (name, item) in enumerate(outputs.items()):
            col = (index % 3) + 1
            row = (index // 3) + 1
            showlegend = True if index == 0 else False

            # Real Plot
            fig.add_trace(
                go.Scatter(
                    y=item['real'],
                    mode='lines',
                    name='Snapshot Real Segment',
                    line=dict(color='Lime'),
                    showlegend=showlegend,
                    fill = 'tozeroy',
                ),
                row=row,
                col=col
            )
            # Simulator Plot
            fig.add_trace(
                go.Scatter(
                    y=item['simulate'],
                    mode='lines',
                    name='Allocator Simulation Segment',
                    line=dict(color='Red'),
                    showlegend=showlegend
                ),
                row=row,
                col=col
            )
            fig.update_xaxes(title_text=name, row=row, col=col)
            fig.update_yaxes(title_text="Memory (GB)", row=row, col=col)
        default_height = 400 * rows
        default_width = 800
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
                font=self.legend_font
            ),
            showlegend=True,
            template=self._plotly_template,
        )
        fig.write_image(
            file=self.get_image_dir(title),
            width=default_width,
            height=default_height
        )
        return fig

    def plot_peak_memory_pre_iteration(self, title: str, data_dir: Union[str, Path]):
        from perf_estimator.snapshot import SnapshotAnalyser
        snapshot_files = filter_files(".pickle", data_dir, True)
        peak_memory = {}
        for index, snapshot in enumerate(snapshot_files):
            p = Path(snapshot)
            model_name = p.name.split(".")[0]
            first_snapshot = SnapshotAnalyser(str(p))
            iteration_peak_mem = first_snapshot.peak_memory_usage_each_iteration()
            segs = []
            traces = []
            for iter_mem in iteration_peak_mem[1:]:
                segs.append(iter_mem['seg'] / 1024 ** 3)
                traces.append(iter_mem['trace'] / 1024 ** 3)
            peak_memory[model_name] = {
                "seg": segs,
                "trace": traces
            }
        return peak_memory

    def random_test_data_processing(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        if str(data_dir) not in self._cache.keys():
            evaluates_files = filter_files("evaluation_result.json", data_dir, False)
            random_result = []
            for eva_file in evaluates_files:
                with open(eva_file, "r") as f:
                    eva_data = json.load(f)
                basic_info = eva_data["train info"]
                basic_info["timestamp"] = "-".join(str(eva_data["config"]["run_id"]).split("-")[:2])
                del eva_data["train info"], eva_data["config"]
                for key, value in eva_data.items():
                    copy_basic_info: dict = copy.deepcopy(basic_info)
                    copy_basic_info["tool"] = key
                    copy_basic_info["2nd oom"] = value.get("2nd verification", {}).get("oom", None)
                    # copy_basic_info["2nd error"] = value.get("2nd verification", {}).get("error", None)
                    copy_basic_info["2nd error"] = value.get("2nd verification", {}).get("error", value["error"])
                    if value["oom"] == False and value["correct_estimation"] == True and copy_basic_info["2nd oom"] == False:
                        accurate_estimation = True
                        save_memory = copy_basic_info['total_gpu_memory'] - (value["memory"] / 1024 ** 3)
                    elif value["oom"] == True and value["correct_estimation"] == True:
                        accurate_estimation = True
                        save_memory = copy_basic_info['total_gpu_memory']
                    else:
                        accurate_estimation = False
                        save_memory = 0

                    copy_basic_info["accurate_estimation"] = accurate_estimation
                    copy_basic_info["save_memory"] = save_memory
                    del value["2nd verification"]
                    copy_basic_info.update(value)
                    random_result.append(copy_basic_info)
            df = pd.DataFrame(random_result)
            df['tool'] = df['tool'].replace(self._name_map)
            df['color'] = df['tool'].replace(self._color_scheme)
            self._cache[str(data_dir)] = df
        return self._cache[str(data_dir)].copy()

    def plot_random_test_cumulative_accurate_correct_estimation(self, title: str, data_dir: Union[str, Path]):
        random_df = self.random_test_data_processing(data_dir)
        random_df["correct_estimation_int"] = random_df["accurate_estimation"].apply(lambda x: 1 if x else 0)

        # 按照时间戳排序
        random_df = random_df.sort_values(by="timestamp")

        random_df['cumulative_correct_estimation'] = random_df.groupby('tool')['correct_estimation_int'].cumsum()
        random_df['cumulative_total'] = random_df.groupby('tool').cumcount() + 1

        random_df['cumulative_correct_estimation_ratio'] = (random_df['cumulative_correct_estimation'] / random_df['cumulative_total']) * 100

        random_df['timestamp'] = pd.to_datetime(random_df['timestamp'], errors='coerce')
        random_df['hour'] = random_df['timestamp'].dt.floor('min')

        fig = px.line(
            random_df,
            x="hour",
            y="cumulative_correct_estimation_ratio",
            color="tool",
            hover_data=["tool", "memory", "oom", "correct_estimation", "2nd oom", "2nd error", "save_memory"],
            labels={"cumulative_correct_estimation_ratio": "Cumulative Correct Estimation Ratio"},
            color_discrete_map = self._color_scheme
        )

        default_width = 800
        default_height = 600
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Probability of Preventing OOM (%)",
            yaxis=dict(showgrid=True, gridcolor='LightGray', zeroline=False, titlefont=self.font),
            xaxis=dict(showgrid=False, titlefont=self.font),
            height=default_height,  # default height
            width=default_width,  # default width
            font=self.font,
            legend=dict(
                title={
                    'text': "Estimator",
                    'font': {**self.legend_font, 'weight': "bold"}
                },
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font
            ),
            showlegend=True,
            template=self._plotly_template,
        )
        fig.write_image(
            file=self.get_image_dir(title),
            width=default_width,
            height=default_height
        )
        return fig

    def plot_random_test_cumulative_correct_estimation(self, title: str, data_dir: Union[str, Path]):
        random_df = self.random_test_data_processing(data_dir)
        random_df["new_correct_estimation_int"] = random_df["correct_estimation"].apply(lambda x: 1 if x else 0)

        random_df = random_df.sort_values(by="timestamp")

        random_df['cumulative_new_correct_estimation'] = random_df.groupby('tool')[
            'new_correct_estimation_int'].cumsum()
        random_df['cumulative_total'] = random_df.groupby('tool').cumcount() + 1

        random_df['cumulative_new_correct_estimation_ratio'] = (random_df['cumulative_new_correct_estimation'] / random_df['cumulative_total']) * 100

        random_df['timestamp'] = pd.to_datetime(random_df['timestamp'], errors='coerce')
        random_df['hour'] = random_df['timestamp'].dt.floor('min')

        fig = px.line(
            random_df,
            x="hour",
            y="cumulative_new_correct_estimation_ratio",
            color="tool",
            hover_data=["tool", "memory", "oom", "correct_estimation", "2nd oom", "2nd error", "save_memory"],
            labels={"cumulative_new_correct_estimation_ratio": "Cumulative New Correct Estimation Ratio"},
            color_discrete_map=self._color_scheme
        )

        default_width = 800
        default_height = 600
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Probability of Preventing OOM (%)",
            yaxis=dict(showgrid=True, gridcolor='LightGray', zeroline=False, titlefont=self.font),
            xaxis=dict(showgrid=False, titlefont=self.font),
            height=default_height,  # default height
            width=default_width,  # default width
            font=self.font,
            legend=dict(
                title={
                    'text': "Estimator",
                    'font': {**self.legend_font, 'weight': "bold"}
                },
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font
            ),
            showlegend=True,
            template=self._plotly_template,
        )
        fig.write_image(
            file=self.get_image_dir(title),
            width=default_width,
            height=default_height
        )
        return fig

    def plot_random_test_memory_saved_bar_chart(self, title: str, data_dir: Union[str, Path]):
        df = self.random_test_data_processing(data_dir)
        # memory_save_acmulative = df.groupby(["tool", "model"])["save_memory"].sum().reset_index()
        memory_save_acmulative = df.groupby(["tool", "model"]).agg(
            save_memory_sum=("save_memory", "sum"),
            save_memory_count=("save_memory", "count")
        ).reset_index()
        memory_save_acmulative["average"] = memory_save_acmulative["save_memory_sum"] / memory_save_acmulative[
            "save_memory_count"]

        memory_save_acmulative = memory_save_acmulative.sort_values(by="save_memory_sum", ascending=False)
        model_order = sorted(df['model'].unique())

        fig = px.bar(
            memory_save_acmulative,
            x="model",
            y="average",
            color="tool",
            barmode='group',
            color_discrete_map = self._color_scheme,
            category_orders={'model': model_order},
        )

        default_width = 800
        default_height = 600
        fig.update_layout(
            xaxis_title="Models",
            yaxis_title="Average Memory Saved (GB)",
            yaxis=dict(showgrid=True, gridcolor='LightGray', zeroline=False, titlefont=self.font),
            xaxis=dict(showgrid=False, titlefont=self.font),
            height=default_height,  # default height
            width=default_width,  # default width
            font=self.font,
            legend=dict(
                title={
                    'text': "Estimator",
                    'font': {**self.legend_font, 'weight': "bold"}
                },
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font
            ),
            showlegend=True,
            template=self._plotly_template,
        )
        fig.write_image(
            file=self.get_image_dir(title),
            width=default_width,
            height=default_height
        )
        return fig

    def plot_cdf_random_test_estimator_performance_score_with_max_gpu_memory(
            self,
            title: str,
            data_dir: Union[str, Path],
            probability_weight: float = 0.7,
            error_weight: float = 0.3
    ):
        df = self.random_test_data_processing(data_dir)
        grouped = df.groupby(['tool', 'model', 'optimiser']).agg(
            success_rate=('correct_estimation', lambda x: (x == True).sum() / len(x)),
            average_error=('error', 'mean'),
        ).reset_index()

        fig = go.Figure()

        for tool_name, group in grouped.groupby('tool'):
            group['combined_score'] = probability_weight * (1 - group['success_rate']) + error_weight * group['average_error']

            sorted_combined_score = np.sort(group['combined_score'].dropna())
            cdf = np.arange(1, len(sorted_combined_score) + 1) / len(sorted_combined_score)

            fig.add_trace(
                go.Scatter(
                    x=sorted_combined_score,
                    y=cdf,
                    mode='lines',
                    name=f'{tool_name}',
                    line=dict(color=self._color_scheme.get(tool_name))
                )
            )

        default_width = 800
        default_height = 600
        fig.update_layout(
            xaxis_title=f"Performance Score({probability_weight}x Success Rate + {error_weight}x Average Error)",
            yaxis_title="CDF",
            yaxis=dict(showgrid=True, gridcolor='LightGray', zeroline=False, titlefont=self.font),
            xaxis=dict(showgrid=False, titlefont=self.font),
            height=default_height,  # default height
            width=default_width,  # default width
            font=self.font,
            legend=dict(
                title={
                    'text': "Estimator",
                    'font': {**self.legend_font, 'weight': "bold"}
                },
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font
            ),
            showlegend=True,
            template=self._plotly_template,
        )
        fig.write_image(
            file=self.get_image_dir(title),
            width=default_width,
            height=default_height
        )
        return fig


    def plot_cdf_random_test_estimator_performance_score_with_estimated_memory(
            self,
            title: str,
            data_dir: Union[str, Path],
            probability_weight: float = 0.7,
            error_weight: float = 0.3
    ):
        df = self.random_test_data_processing(data_dir)
        # df['2nd oom'] = df['2nd oom'].infer_objects(copy=False)
        # df['2nd oom'] = df['2nd oom'].astype(bool)

        # grouped = df.groupby(['tool', 'model', "optimiser"]).agg(
        #     second_oom_success_rate=('2nd oom', lambda x: (~x).sum() / len(x)),  # ~x 表示取反，即 False 的数量
        #     average_second_error=('2nd error', 'mean'),
        # ).reset_index()
        grouped = df.groupby(['tool', 'model', "optimiser"]).agg(
            second_oom_success_rate=('accurate_estimation', lambda x: (x==True).sum() / len(x)),  # ~x 表示取反，即 False 的数量
            average_second_error=('2nd error', 'mean'),
        ).reset_index()

        fig = go.Figure()

        for tool_name, group in grouped.groupby('tool'):
            group['combined_score'] = probability_weight * (1 - group['second_oom_success_rate']) + error_weight * group['average_second_error']

            sorted_combined_score = np.sort(group['combined_score'].dropna())
            cdf = np.arange(1, len(sorted_combined_score) + 1) / len(sorted_combined_score)

            # 为每个工具绘制 CDF
            fig.add_trace(go.Scatter(
                x=sorted_combined_score,
                y=cdf,
                mode='lines',
                name=f'{tool_name}',
                line=dict(color=self._color_scheme.get(tool_name))
            ))

        default_width = 800
        default_height = 600
        fig.update_layout(
            xaxis_title=f"Performance Score({probability_weight}x Success Rate + {error_weight}x Average Error)",
            yaxis_title="CDF",
            yaxis=dict(showgrid=True, gridcolor='LightGray', zeroline=False, titlefont=self.font),
            xaxis=dict(showgrid=False, titlefont=self.font),
            height=default_height,  # default height
            width=default_width,  # default width
            font=self.font,
            legend=dict(
                title={
                    'text': "Estimator",
                    'font': {**self.legend_font, 'weight': "bold"}
                },
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
                bordercolor="Black",
                borderwidth=2,
                font=self.legend_font
            ),
            showlegend=True,
            template=self._plotly_template,
        )
        fig.write_image(
            file=self.get_image_dir(title),
            width=default_width,
            height=default_height
        )
        return fig



    def plot_random_test_sankey_diagram(self, title: str, data_dir: Union[str, Path]):
        df = self.random_test_data_processing(data_dir)
        df['tool'] = df['tool'].str.capitalize()

        # 创建各部分的标签和源、目标列表
        labels = [
            'Total Tests',
            'Tool: Solution', 'Tool: Dnnmem', 'Tool: Schedtune', 'Tool: Llmem',
            'Correct Estimation: True', 'Correct Estimation: False',
            '2nd OOM: True', '2nd OOM: False',
            'Tool Final: Solution', 'Tool Final: Dnnmem', 'Tool Final: Schedtune', 'Tool Final: Llmem'
        ]

        total_tests = len(df)

        # 第二段：通过Tool Name分流
        tool_counts = df['tool'].value_counts()

        # 第三段：预测准确correct_estimation
        tool_correct_counts = df.groupby(['tool', 'correct_estimation']).size().unstack(fill_value=0)

        # 第四段：'2nd oom'
        tool_oom_counts = df.groupby(['tool', 'correct_estimation', '2nd oom']).size().unstack(fill_value=0)

        # 创建源、目标和流量列表
        source = []
        target = []
        value = []

        # 从Total Tests到Tool Name的流
        tool_names = tool_counts.index.tolist()
        for tool in tool_names:
            source.append(0)  # Total Tests
            target.append(labels.index(f'Tool: {tool}'))
            value.append(tool_counts[tool])

        # 从Tool Name到Correct Estimation的流
        for tool in tool_names:
            tool_index = labels.index(f'Tool: {tool}')
            if tool in tool_correct_counts.index:
                if True in tool_correct_counts.columns:
                    source.append(tool_index)
                    target.append(labels.index('Correct Estimation: True'))
                    value.append(tool_correct_counts.loc[tool, True])
                if False in tool_correct_counts.columns:
                    source.append(tool_index)
                    target.append(labels.index('Correct Estimation: False'))
                    value.append(tool_correct_counts.loc[tool, False])

        # 从Correct Estimation到2nd OOM的流
        for tool in tool_names:
            for correct in [True, False]:
                if (tool, correct) in tool_oom_counts.index:
                    correct_index = labels.index(f'Correct Estimation: {correct}')
                    if True in tool_oom_counts.columns:
                        source.append(correct_index)
                        target.append(labels.index('2nd OOM: True'))
                        value.append(tool_oom_counts.loc[(tool, correct), True])
                    if False in tool_oom_counts.columns:
                        source.append(correct_index)
                        target.append(labels.index('2nd OOM: False'))
                        value.append(tool_oom_counts.loc[(tool, correct), False])

        # 从2nd OOM到最终Tool Name的流
        for tool in tool_names:
            for correct in [True, False]:
                for oom in [True, False]:
                    if (tool, correct) in tool_oom_counts.index and oom in tool_oom_counts.columns:
                        oom_index = labels.index(f'2nd OOM: {oom}')
                        tool_final_index = labels.index(f'Tool Final: {tool}')
                        source.append(oom_index)
                        target.append(tool_final_index)
                        value.append(tool_oom_counts.loc[(tool, correct), oom])

        # 创建Sankey图
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])

        fig.update_layout(title_text="Sankey Diagram of Test Flow", font_size=10)
        fig.show()














