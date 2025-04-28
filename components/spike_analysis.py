"""
Spike analysis component for the visualization app.
"""

from functools import partial

import numpy as np
import pandas as pd
import panel as pn
from bokeh.layouts import gridplot
from bokeh.models import BoxZoomTool, ColumnDataSource, HoverTool, Span, WheelZoomTool
from bokeh.plotting import figure
from scipy.signal import savgol_filter
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from umap import UMAP

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.pipeline_util.s3 import get_public_representative_spikes


class RawSpikeAnalysis:
    """Handles spike waveform analysis and visualization."""

    def __init__(self, df_meta: pd.DataFrame, main_app):
        """Initialize with metadata dataframe."""
        self.main_app = main_app
        self.df_meta = df_meta

        # Load extracted raw spike data
        self.df_spikes = get_public_representative_spikes()
        self.extract_from_options = self.df_spikes.index.get_level_values(1).unique()

    def _normalize(self, x, idx_range_to_norm=None):
        """Normalize data within a specified range."""
        x0 = x if idx_range_to_norm is None else x[:, idx_range_to_norm]
        min_vals = np.min(x0, axis=1, keepdims=True)
        range_vals = np.ptp(x0, axis=1, keepdims=True)
        return (x - min_vals) / range_vals

    def extract_representative_spikes(
        self,
        extract_from,
        if_normalize_v: bool = True,
        normalize_window_v: tuple = (-2, 4),
        if_normalize_dvdt: bool = True,
        normalize_window_dvdt: tuple = (-2, 0),
        if_smooth_dvdt: bool = True,
        filtered_df_meta: pd.DataFrame = None,
    ):
        """Extract and process representative spike waveforms."""
        # Get the waveforms
        df_waveforms = self.df_spikes.query("extract_from == @extract_from")
        
        # Filter by filtered_df_meta
        if filtered_df_meta is not None:
            df_waveforms = df_waveforms.query(
                "ephys_roi_id in @filtered_df_meta.ephys_roi_id.values"
            )

        if len(df_waveforms) == 0:
            raise ValueError(f"No waveforms found for extract_from={extract_from}")

        t = df_waveforms.columns.values.T
        v = df_waveforms.values
        dvdt = np.gradient(v, t, axis=1)

        # Normalize the dvdt
        if if_normalize_dvdt:
            dvdt = self._normalize(
                dvdt,
                idx_range_to_norm=np.where(
                    (t >= normalize_window_dvdt[0]) & (t <= normalize_window_dvdt[1])
                )[0],
            )

        if if_smooth_dvdt:
            dvdt = savgol_filter(dvdt, window_length=5, polyorder=3, axis=1)

        dvdt_max_idx = np.argmax(dvdt, axis=1)
        max_shift_right = dvdt_max_idx.max() - dvdt_max_idx.min()

        # Calculate new time array that spans all possible shifts
        dt = t[1] - t[0]
        t_dvdt = -dvdt_max_idx.max() * dt + np.arange(len(t) + max_shift_right) * dt

        # Create new dvdt array with NaN padding
        new_dvdt = np.full((dvdt.shape[0], len(t_dvdt)), np.nan)

        # For each cell, place its dvdt trace in the correct position
        for i, (row, peak_idx) in enumerate(zip(dvdt, dvdt_max_idx)):
            start_idx = dvdt_max_idx.max() - peak_idx  # Align the max_index
            new_dvdt[i, start_idx : start_idx + len(row)] = row

        # Normalize the v
        if if_normalize_v:
            idx_range_to_norm = np.where(
                (t >= normalize_window_v[0]) & (t <= normalize_window_v[1])
            )[0]
            v = self._normalize(v, idx_range_to_norm)

        self.normalize_window_v = normalize_window_v
        self.normalize_window_dvdt = normalize_window_dvdt

        # Create dictionary with ephys_roi_id as keys
        df_v_norm = pd.DataFrame(v, index=df_waveforms.index.get_level_values(0), columns=t)
        df_dvdt_norm = pd.DataFrame(
            new_dvdt, index=df_waveforms.index.get_level_values(0), columns=t_dvdt
        )

        return df_v_norm, df_dvdt_norm

    def create_plot_controls(self) -> dict:
        """Create control widgets for spike analysis."""
        controls = {
            "extract_from": pn.widgets.Select(
                name="Extract spikes from",
                options=sorted(self.extract_from_options.tolist()),
                value="long_square_rheo, min",
                sizing_mode="stretch_width",
            ),
            "dim_reduction_method": pn.widgets.Select(
                name="Dimensionality Reduction Method",
                options=["PCA", "UMAP"],
                value="PCA",
                sizing_mode="stretch_width",
            ),
            "spike_range": pn.widgets.RangeSlider(
                name="Spike Analysis Range (ms)",
                start=-5,
                end=10,
                value=(-3, 6),
                step=0.5,
                sizing_mode="stretch_width",
            ),
            "normalize_window_v": pn.widgets.RangeSlider(
                name="V Normalization Window",
                start=-4,
                end=7,
                value=(-2, 4),
                step=0.5,
                sizing_mode="stretch_width",
            ),
            "normalize_window_dvdt": pn.widgets.RangeSlider(
                name="dV/dt Normalization Window",
                start=-3,
                end=6,
                value=(-2, 0),
                step=0.5,
                sizing_mode="stretch_width",
            ),
            "n_clusters": pn.widgets.IntSlider(
                name="Number of Clusters",
                start=2,
                end=5,
                value=2,
                step=1,
                sizing_mode="stretch_width",
            ),
            "if_show_cluster_on_retro": pn.widgets.Checkbox(
                name="Show type color for Retro",
                value=False,
                sizing_mode="stretch_width",
            ),
            "marker_size": pn.widgets.IntSlider(
                name="Marker Size",
                start=5,
                end=20,
                value=13,
                step=1,
                sizing_mode="stretch_width",
            ),
            "alpha_slider": pn.widgets.FloatSlider(
                name="Alpha",
                start=0.1,
                end=1.0,
                value=0.3,
                step=0.1,
                sizing_mode="stretch_width",
            ),
            "plot_width": pn.widgets.IntSlider(
                name="Plot Width",
                start=200,
                end=800,
                value=550,
                step=50,
                sizing_mode="stretch_width",
            ),
            "plot_height": pn.widgets.IntSlider(
                name="Plot Height",
                start=200,
                end=800,
                value=550,
                step=50,
                sizing_mode="stretch_width",
            ),
            "font_size": pn.widgets.IntSlider(
                name="Font Size",
                start=8,
                end=24,
                value=12,
                step=1,
                sizing_mode="stretch_width",
            ),
        }
        return controls

    def perform_dim_reduction_clustering(
        self, df_v_norm: pd.DataFrame, n_clusters: int = 2, method: str = "PCA"
    ):
        """
        Perform dimensionality reduction and K-means clustering on the voltage traces.

        Parameters:
            df_v_norm : pd.DataFrame
                Normalized voltage traces
            n_clusters : int
                Number of clusters for K-means
            method : str
                Dimensionality reduction method ("PCA" or "UMAP")
        """
        v = df_v_norm.values

        if method == "PCA":
            # Perform PCA
            reducer = PCA()
            v_proj = reducer.fit_transform(v)
            n_components = 5
            columns = [f"PCA{i}" for i in range(1, n_components + 1)]
        else:  # UMAP
            # Perform UMAP
            reducer = UMAP(n_components=2, random_state=42)
            v_proj = reducer.fit_transform(v)
            n_components = 2
            columns = [f"UMAP{i}" for i in range(1, n_components + 1)]

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(v_proj[:, :2])

        # Calculate metrics
        silhouette_avg = silhouette_score(v_proj[:, :2], clusters)
        metrics = {
            "silhouette_avg": silhouette_avg,
        }

        # Save data
        df_v_proj = pd.DataFrame(v_proj[:, :n_components], index=df_v_norm.index, columns=columns)

        # Add cluster information to df_v_norm
        clusters_df = pd.DataFrame(clusters, index=df_v_norm.index, columns=["cluster_id"])
        self.df_meta = self.df_meta[
            [col for col in self.df_meta.columns if col != "cluster_id"]
        ].merge(clusters_df, on="ephys_roi_id", how="left")
        df_v_proj = df_v_proj.merge(clusters_df, on="ephys_roi_id", how="left")
        df_v_proj = df_v_proj.merge(
            self.df_meta[
                [
                    "Date_str",
                    "ephys_roi_id",
                    "injection region",
                    "cell_summary_url",
                    "jem-id_cell_specimen",
                ]
            ],
            on="ephys_roi_id",
            how="left",
        )

        return df_v_proj, clusters, reducer, metrics

    def create_tooltips(
        self,
    ):
        """Create tooltips for the hover tool."""

        tooltips = """
             <div style="text-align: left; flex: auto; white-space: nowrap; margin: 0 10px;
                       border: 2px solid black; padding: 10px;">
                    <span style="font-size: 17px;">
                        <b>@Date_str, @{injection region}, @{ephys_roi_id},
                            @{jem-id_cell_specimen}</b><br>
                    </span>
                    <img src="@cell_summary_url{safe}" alt="Cell Summary"
                         style="width: 800px; height: auto;">
             </div>
             """

        return tooltips

    # Add callback to update ephys_roi_id on point tap
    def update_ephys_roi_id(self, df, attr, old, new):
        if new:
            selected_index = new[0]
            ephys_roi_id = str(int(df["ephys_roi_id"][selected_index]))
            # Update the data holder's ephys_roi_id
            if hasattr(self.main_app, "data_holder"):
                self.main_app.data_holder.ephys_roi_id_selected = ephys_roi_id

    def create_raw_PCA_plots(
        self,
        df_v_norm: pd.DataFrame,
        df_dvdt_norm: pd.DataFrame,
        n_clusters: int = 2,
        alpha: float = 0.3,
        width: int = 400,
        height: int = 400,
        font_size: int = 12,
        marker_size: int = 10,
        if_show_cluster_on_retro: bool = True,
        spike_range: tuple = (-4, 7),
        dim_reduction_method: str = "PCA",
            ) -> gridplot:
        """Create plots for spike analysis including dimensionality reduction and clustering."""
                # Filter data based on spike_range
        df_v_norm = df_v_norm.loc[
            :, (df_v_norm.columns >= spike_range[0]) & (df_v_norm.columns <= spike_range[1])
        ]
        df_dvdt_norm = df_dvdt_norm.loc[
            :, (df_dvdt_norm.columns >= spike_range[0]) & (df_dvdt_norm.columns <= spike_range[1])
        ]

        # Perform dimensionality reduction and clustering
        df_v_proj, clusters, reducer, metrics = self.perform_dim_reduction_clustering(
            df_v_norm, n_clusters, dim_reduction_method
        )
        cluster_colors = ["black", "darkgray", "darkblue", "cyan", "darkorange"][:n_clusters]

        # Common plot settings
        plot_settings = dict(width=width, height=height)

        # Create figures
        p1 = figure(
            x_axis_label=f"{dim_reduction_method}1",
            y_axis_label=f"{dim_reduction_method}2",
            tools="pan,reset,tap,wheel_zoom,box_select,lasso_select",
            **plot_settings,
        )
        p2 = figure(
            title=f"Raw Vm, normalized between {self.normalize_window_v[0]} "
            f"to {self.normalize_window_v[1]} ms",
            x_axis_label="Time (ms)",
            y_axis_label="V",
            x_range=(spike_range[0] - 0.1, spike_range[1] + 0.1),
            tools="pan,reset,tap,wheel_zoom,box_select,lasso_select",
            **plot_settings,
        )
        p3 = figure(
            title=f"dV/dt, normalized betwen {self.normalize_window_dvdt[0]} "
            f"to {self.normalize_window_dvdt[1]} ms",
            x_axis_label="Time (ms)",
            y_axis_label="dV/dt",
            x_range=(spike_range[0] - 0.1, spike_range[1] + 0.1),
            tools="pan,reset,tap,wheel_zoom,box_select,lasso_select",
            **plot_settings,
        )

        # Update font sizes after figure creation
        for p in [p1, p2, p3]:
            # Set the font sizes for the title and axis labels
            p.title.text_font_size = f"{font_size+2}pt"
            p.xaxis.axis_label_text_font_size = f"{font_size+2}pt"
            p.yaxis.axis_label_text_font_size = f"{font_size+2}pt"

            # Set the font sizes for the major tick labels on the axes
            p.xaxis.major_label_text_font_size = f"{font_size}pt"
            p.yaxis.major_label_text_font_size = f"{font_size}pt"

            # Set legend font size if legend exists
            if p.legend:
                p.legend.label_text_font_size = f"{font_size}pt"

        # -- Plot PCA scatter with contours --
        # Create a single ColumnDataSource for all clusters
        # If injection region is not "Non-Retro", set color to None
        scatter_list_p1 = []

        for i in df_v_proj["cluster_id"].unique():
            # Add dots
            querystr = "cluster_id == @i"
            group_label = f"Cluster {i+1}"
            if not if_show_cluster_on_retro:
                querystr += " and `injection region` == 'Non-Retro'"
                group_label += " (Non-Retro)"

            group_label += f", n={df_v_proj.query(querystr).shape[0]}"

            source = ColumnDataSource(df_v_proj.query(querystr))
            scatter_list_p1.append(
                p1.scatter(
                    x=f"{dim_reduction_method}1",
                    y=f"{dim_reduction_method}2",
                    source=source,
                    size=marker_size,
                    color=cluster_colors[i],
                    alpha=alpha,
                    legend_label=group_label,
                    hover_color="blue",
                    selection_color="blue",
                )
            )

            # Attach the callback to the selection changes
            source.selected.on_change("indices", partial(self.update_ephys_roi_id, source.data))

            # Add contours
            values = (
                df_v_proj.query("cluster_id == @i")
                .loc[:, [f"{dim_reduction_method}1", f"{dim_reduction_method}2"]]
                .values
            )
            mean = np.mean(values, axis=0)
            cov = np.cov(values.T)
            x, y = np.mgrid[
                values[:, 0].min() - 0.5 : values[:, 0].max() + 0.5 : 100j,
                values[:, 1].min() - 0.5 : values[:, 1].max() + 0.5 : 100j,
            ]
            pos = np.dstack((x, y))
            rv = multivariate_normal(mean, cov)
            z = rv.pdf(pos)
            add_counter(p1, x, y, z, levels=3, line_color=cluster_colors[i], alpha=1)

        # Add metrics to the plot
        p1.title.text = (
            f"{dim_reduction_method} + K-means Clustering (n_clusters = {n_clusters})\n"
            f"Silhouette Score: {metrics['silhouette_avg']:.3f}\n"
        )
        p1.toolbar.active_scroll = p1.select_one(WheelZoomTool)

        # Add vertical lines for normalization windows
        p2.add_layout(
            Span(
                location=self.normalize_window_v[0],
                dimension="height",
                line_color="blue",
                line_dash="dashed",
                line_width=2,
            )
        )
        p2.add_layout(
            Span(
                location=self.normalize_window_v[1],
                dimension="height",
                line_color="blue",
                line_dash="dashed",
                line_width=2,
            )
        )
        p3.add_layout(
            Span(
                location=self.normalize_window_dvdt[0],
                dimension="height",
                line_color="blue",
                line_dash="dashed",
                line_width=2,
            )
        )
        p3.add_layout(
            Span(
                location=self.normalize_window_dvdt[1],
                dimension="height",
                line_color="blue",
                line_dash="dashed",
                line_width=2,
            )
        )

        # Add boxzoomtool to p2 and p3
        box_zoom_x = BoxZoomTool(dimensions="auto")
        p2.add_tools(box_zoom_x)
        p2.toolbar.active_drag = box_zoom_x
        box_zoom_x = BoxZoomTool(dimensions="auto")
        p3.add_tools(box_zoom_x)
        p3.toolbar.active_drag = box_zoom_x

        # Plot voltage and dV/dt traces
        for i in range(n_clusters):
            query_str = "cluster_id == @i"
            group_label = f"Cluster {i+1}"
            if not if_show_cluster_on_retro:
                query_str += " and `injection region` == 'Non-Retro'"
                group_label += " (Non-Retro)"
            group_label += f", n={df_v_proj.query(query_str).shape[0]}"
            ephys_roi_ids = df_v_proj.query(query_str).ephys_roi_id.tolist()

            # Common line properties
            line_props = {
                "alpha": alpha,
                "hover_line_color": "blue",
                "hover_line_alpha": 1.0,
                "hover_line_width": 4,
                "selection_line_color": "blue",
                "selection_line_alpha": 1.0,
                "selection_line_width": 4,
            }
            # Plot voltage traces
            df_this = df_v_norm.query("ephys_roi_id in @ephys_roi_ids")
            source = ColumnDataSource(
                {
                    "xs": [df_v_norm.columns.values] * len(df_this),
                    "ys": df_this.values.tolist(),
                    "ephys_roi_id": ephys_roi_ids,
                }
            )

            p2.multi_line(
                source=source,
                xs="xs",
                ys="ys",
                color=cluster_colors[i],
                **line_props,
                legend_label=group_label,
            )

            # Plot dV/dt traces
            df_this = df_dvdt_norm.query("ephys_roi_id in @ephys_roi_ids")
            source = ColumnDataSource(
                {
                    "xs": [df_dvdt_norm.columns.values] * len(df_this),
                    "ys": df_this.values.tolist(),
                    "ephys_roi_id": ephys_roi_ids,
                }
            )
            p3.multi_line(
                source=source,
                xs="xs",
                ys="ys",
                color=cluster_colors[i],
                **line_props,
                legend_label=group_label,
            )

        # Add region cluster_colors to the all plots
        for region in self.df_meta["injection region"].unique():
            if region == "Non-Retro":
                continue
            roi_ids = self.df_meta.query("`injection region` == @region").ephys_roi_id.tolist()
            legend_label = f"{region}, n={len(roi_ids)}"

            source = ColumnDataSource(df_v_proj.query("ephys_roi_id in @roi_ids"))
            scatter_list_p1.append(
                p1.scatter(
                    x=f"{dim_reduction_method}1",
                    y=f"{dim_reduction_method}2",
                    source=source,
                    color=REGION_COLOR_MAPPER[region],
                    alpha=0.8,
                    size=marker_size,
                    legend_label=legend_label,
                )
            )

            # Attach the callback to the selection changes
            source.selected.on_change("indices", partial(self.update_ephys_roi_id, source.data))

            ys = df_v_norm.query("ephys_roi_id in @roi_ids").values

            # Common line properties
            line_props = {
                "hover_line_color": "blue",
                "hover_line_alpha": 1.0,
                "hover_line_width": 4,
                "selection_line_color": "blue",
                "selection_line_alpha": 1.0,
                "selection_line_width": 4,
            }
            p2.multi_line(
                xs=[df_v_norm.query("ephys_roi_id in @roi_ids").columns.values] * ys.shape[0],
                ys=ys.tolist(),
                color=REGION_COLOR_MAPPER[region],
                alpha=0.8,
                legend_label=legend_label,
                **line_props,
            )
            ys = df_dvdt_norm.query("ephys_roi_id in @roi_ids").values
            p3.multi_line(
                xs=[df_dvdt_norm.query("ephys_roi_id in @roi_ids").columns.values] * ys.shape[0],
                ys=ys.tolist(),
                color=REGION_COLOR_MAPPER[region],
                alpha=0.8,
                legend_label=legend_label,
                **line_props,
            )

        # Add tooltips
        # Add renderers like this to solve bug like this:
        #   File "/Users/han.hou/miniconda3/envs/patch-seq/lib/python3.10/
        # site-packages/panel/io/location.py", line 57, in _get_location_params
        #     params['pathname'], search = uri.split('?')
        # ValueError: too many values to unpack (expected 2)
        # 2025-04-09 00:03:04,658 500 GET /patchseq_panel_viz??? (::1) 8541.01ms
        hovertool = HoverTool(
            tooltips=self.create_tooltips(),
            renderers=scatter_list_p1,
        )
        p1.add_tools(hovertool)

        hovertool = HoverTool(
            tooltips=[("ephys_roi_id", "@ephys_roi_id")],
            attachment="right",  # Fix tooltip to the right of the plot
        )
        p2.add_tools(hovertool)
        p3.add_tools(hovertool)

        for p in [p1, p2, p3]:
            p.legend.ncols = 2
            p.legend.background_fill_alpha = 0.5
            p.legend.location = "bottom_center"
            p.legend.click_policy = "hide"
            p.legend.orientation = "horizontal"

        # Create grid layout with independent axes
        layout = gridplot([[p1, p2, p3]], toolbar_location="right", merge_tools=False)
        
        return layout


def add_counter(p, x, y, z, levels=5, line_color="blue", alpha=0.5, line_width=2):
    """
    Add contour lines to a Bokeh figure.

    This function uses Matplotlib's contour function to compute contour lines
    based on a grid defined by x, y, and corresponding values z. The contour lines
    are then extracted and added to the provided Bokeh plot using the multi_line glyph.

    Parameters:
        p : bokeh.plotting.figure.Figure
            The Bokeh figure to which the contour lines will be added.
        x, y : 2D arrays
            The grid arrays for the x and y coordinates (e.g., generated by numpy.meshgrid).
        z : 2D array
            The array of values over the grid defined by x and y.
        levels : int, optional
            The number of contour levels to compute (default is 5).
        line_color : str, optional
            The color to use for the contour lines (default is "blue").
        alpha : float, optional
            The transparency level of the contour lines (default is 0.5).
        line_width : int, optional
            The width of the contour lines (default is 2).
    """
    import matplotlib.pyplot as plt

    # Compute contour lines using Matplotlib
    plt.figure()  # create a temporary figure for calculating contours
    contour_set = plt.contour(x, y, z, levels=levels)
    plt.close()  # close the figure; we're only interested in the data

    xs_list, ys_list = [], []
    alphas = []
    # Use the 'allsegs' attribute which contains a list of segment lists
    for i, segs in enumerate(contour_set.allsegs):
        # Calculate decreasing alpha for each contour level
        level_alpha = alpha * (i / len(contour_set.allsegs))
        for seg in segs:
            xs_list.append(seg[:, 0].tolist())
            ys_list.append(seg[:, 1].tolist())
            alphas.append(level_alpha)

    # Plot the extracted contour lines on the Bokeh figure with varying alpha
    renderer = p.multi_line(
        xs=xs_list,
        ys=ys_list,
        line_color=line_color,
        line_alpha=alphas,
        line_width=line_width,
        name="contour_lines",  # Add a name for easier reference
        level="underlay",  # Place contour lines under other glyphs
    )

    # Make contour lines non-interactive
    renderer.nonselection_glyph = None  # Disable selection
    renderer.selection_glyph = None  # Disable selection
    renderer.hover_glyph = None  # Disable hover
    renderer.propagate_hover = False  # Prevent hover events from propagating
