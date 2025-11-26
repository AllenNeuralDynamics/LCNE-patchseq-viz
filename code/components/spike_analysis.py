"""
Spike analysis component for the visualization app.
"""

from functools import partial

import numpy as np
import pandas as pd
import panel as pn
from bokeh.layouts import gridplot
from bokeh.models import BoxZoomTool, ColumnDataSource, CustomJS, HoverTool, Span, WheelZoomTool
from bokeh.plotting import figure
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
try:  # UMAP does not work in Hugging Face Spaces
    from umap import UMAP
except:
    UMAP = None

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.pipeline_util.s3 import get_public_representative_spikes

from components.utils.svg_export import export_figures_to_svg_zip


class RawSpikeAnalysis:
    """Handles spike waveform analysis and visualization."""

    def __init__(self, df_meta: pd.DataFrame, main_app):
        """Initialize with metadata dataframe."""
        self.main_app = main_app
        self.df_meta = df_meta
        self._latest_figures = {}

        # Load extracted raw spike data
        self.df_spikes = get_public_representative_spikes()
        self.extract_from_options = self.df_spikes.index.get_level_values(1).unique()

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
        df_v_phase_norm: pd.DataFrame | None = None,
        df_dvdt_phase_norm: pd.DataFrame | None = None,
        df_v_unnorm: pd.DataFrame = None,
        df_dvdt_unnorm: pd.DataFrame = None,
        n_clusters: int = 2,
        alpha: float = 0.3,
        width: int = 400,
        height: int = 400,
        font_size: int = 12,
        marker_size: int = 10,
        if_show_cluster_on_retro: bool = True,
        spike_range: tuple = (-4, 7),
        dim_reduction_method: str = "PCA",
        normalize_window_v: tuple = (-2, 4),
        normalize_window_dvdt: tuple = (-2, 0),
    ) -> gridplot:
        """Create plots for spike analysis including dimensionality reduction and clustering."""
        # Filter data based on spike_range
        df_v_norm = df_v_norm.loc[
            :, (df_v_norm.columns >= spike_range[0]) & (df_v_norm.columns <= spike_range[1])
        ]
        df_dvdt_norm = df_dvdt_norm.loc[
            :, (df_dvdt_norm.columns >= spike_range[0]) & (df_dvdt_norm.columns <= spike_range[1])
        ]

        if df_v_phase_norm is not None:
            df_v_phase_norm = df_v_phase_norm.loc[
                :, (df_v_phase_norm.columns >= spike_range[0])
                & (df_v_phase_norm.columns <= spike_range[1])
            ]
        if df_dvdt_phase_norm is not None:
            df_dvdt_phase_norm = df_dvdt_phase_norm.loc[
                :, (df_dvdt_phase_norm.columns >= spike_range[0])
                & (df_dvdt_phase_norm.columns <= spike_range[1])
            ]
        
        # Filter unnormalized data if provided
        if df_v_unnorm is not None:
            df_v_unnorm = df_v_unnorm.loc[
                :, (df_v_unnorm.columns >= spike_range[0]) & (df_v_unnorm.columns <= spike_range[1])
            ]
        if df_dvdt_unnorm is not None:
            df_dvdt_unnorm = df_dvdt_unnorm.loc[
                :, (df_dvdt_unnorm.columns >= spike_range[0]) & (df_dvdt_unnorm.columns <= spike_range[1])
            ]

        # Perform dimensionality reduction and clustering
        df_v_proj, clusters, reducer, metrics = self.perform_dim_reduction_clustering(
            df_v_norm, n_clusters, dim_reduction_method
        )
        cluster_colors = ["black", "darkgray", "darkblue", "cyan", "darkorange"][:n_clusters]

        # Common plot settings
        plot_settings = dict(width=width, height=height)
        legend_groups = {}

        def register_renderer(label, renderer):
            if not label or renderer is None:
                return
            legend_groups.setdefault(label, []).append(renderer)

        def add_timeseries_mean_sem(fig, df_values, color, label):
            if df_values is None or df_values.empty:
                return
            mean = df_values.mean(axis=0)
            if mean.isna().all():
                return
            sem = df_values.sem(axis=0).fillna(0)
            x_vals = pd.to_numeric(mean.index, errors="coerce")
            valid_mask = ~(np.isnan(x_vals) | np.isnan(mean.values))
            if not valid_mask.any():
                return
            x_vals = x_vals[valid_mask]
            mean_vals = mean.values[valid_mask]
            sem_vals = sem.values[valid_mask]
            legend_label = f"{label} (mean±SEM)"
            source = ColumnDataSource(
                {
                    "x": x_vals,
                    "mean": mean_vals,
                    "upper": mean_vals + sem_vals,
                    "lower": mean_vals - sem_vals,
                }
            )
            band = fig.varea(
                x="x",
                y1="lower",
                y2="upper",
                source=source,
                fill_color=color,
                fill_alpha=0.15,
                level="underlay",
            )
            register_renderer(legend_label, band)
            line = fig.line(
                x="x",
                y="mean",
                source=source,
                color=color,
                line_width=3,
                legend_label=legend_label,
            )
            register_renderer(legend_label, line)

        def add_phase_mean_sem(fig, df_v_values, df_dvdt_values, color, label, n_bins=100):
            if (
                df_v_values is None
                or df_dvdt_values is None
                or df_v_values.empty
                or df_dvdt_values.empty
            ):
                return
            v_vals = df_v_values.to_numpy().astype(float, copy=False).ravel()
            dvdt_vals = df_dvdt_values.to_numpy().astype(float, copy=False).ravel()
            finite_mask = np.isfinite(v_vals) & np.isfinite(dvdt_vals)
            if not finite_mask.any():
                return
            v_vals = v_vals[finite_mask]
            dvdt_vals = dvdt_vals[finite_mask]
            v_min, v_max = np.min(v_vals), np.max(v_vals)
            if v_min == v_max:
                return
            bin_edges = np.linspace(v_min, v_max, n_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            def render_segment(mask, segment_label, line_dash):
                x_seg = v_vals[mask]
                y_seg = dvdt_vals[mask]
                if x_seg.size < 5:
                    return
                bin_indices = np.digitize(x_seg, bin_edges) - 1
                valid = (bin_indices >= 0) & (bin_indices < n_bins)
                if not valid.any():
                    return
                bin_indices = bin_indices[valid]
                y_seg = y_seg[valid]
                means = []
                sems = []
                centers = []
                for b_idx in range(n_bins):
                    bin_mask = bin_indices == b_idx
                    count = np.count_nonzero(bin_mask)
                    if count < 3:
                        continue
                    values = y_seg[bin_mask]
                    centers.append(bin_centers[b_idx])
                    means.append(np.mean(values))
                    sems.append(np.std(values, ddof=1) / np.sqrt(count) if count > 1 else 0.0)
                if len(centers) < 2:
                    return
                centers = np.array(centers)
                means = np.array(means)
                sems = np.array(sems)
                legend_label = f"{label} ({segment_label} mean±SEM)"
                band_source = ColumnDataSource(
                    {
                        "x": np.concatenate([centers, centers[::-1]]),
                        "y": np.concatenate([means + sems, (means - sems)[::-1]]),
                    }
                )
                band = fig.patch(
                    x="x",
                    y="y",
                    source=band_source,
                    fill_color=color,
                    fill_alpha=0.1,
                    line_alpha=0,
                    level="underlay",
                )
                register_renderer(legend_label, band)
                line_source = ColumnDataSource({"x": centers, "y": means})
                line = fig.line(
                    x="x",
                    y="y",
                    source=line_source,
                    color=color,
                    line_width=3,
                    line_dash=line_dash,
                    legend_label=legend_label,
                )
                register_renderer(legend_label, line)

            render_segment(dvdt_vals >= 0, "dV/dt > 0", "solid")
            render_segment(dvdt_vals < 0, "dV/dt < 0", "dashed")

        plots = self._init_spike_subplots(
            dim_reduction_method,
            spike_range,
            normalize_window_v,
            normalize_window_dvdt,
            plot_settings,
        )
        p_embedding = plots["embedding"]
        p_vm = plots["vm"]
        p_dvdt = plots["dvdt"]
        p_phase_norm = plots["phase_norm"]
        p_phase = plots["phase"]

        self._style_subplots(plots.values(), font_size)

        phase_norm_v = df_v_phase_norm if df_v_phase_norm is not None else df_v_norm
        phase_norm_dvdt = (
            df_dvdt_phase_norm if df_dvdt_phase_norm is not None else df_dvdt_norm
        )

        # -- Plot PCA scatter with contours --
        # Create a single ColumnDataSource for all clusters
        # If injection region is not "Non-Retro", set color to None
        scatter_renderers = []

        for i in df_v_proj["cluster_id"].unique():
            # Add dots
            querystr = "cluster_id == @i"
            group_label = f"Cluster {i+1}"
            if not if_show_cluster_on_retro:
                querystr += " and `injection region` == 'Non-Retro'"
                group_label += " (Non-Retro)"

            group_label += f", n={df_v_proj.query(querystr).shape[0]}"

            source = ColumnDataSource(df_v_proj.query(querystr))
            scatter = p_embedding.scatter(
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
            scatter_renderers.append(scatter)
            register_renderer(group_label, scatter)

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
            add_counter(p_embedding, x, y, z, levels=3, line_color=cluster_colors[i], alpha=1)

        # Add metrics to the plot
        p_embedding.title.text = (
            f"{dim_reduction_method} + K-means Clustering (n_clusters = {n_clusters})\n"
            f"Silhouette Score: {metrics['silhouette_avg']:.3f}\n"
        )
        p_embedding.toolbar.active_scroll = p_embedding.select_one(WheelZoomTool)

        # Add vertical lines for normalization windows
        p_vm.add_layout(
            Span(
                location=normalize_window_v[0],
                dimension="height",
                line_color="blue",
                line_dash="dashed",
                line_width=2,
            )
        )
        p_vm.add_layout(
            Span(
                location=normalize_window_v[1],
                dimension="height",
                line_color="blue",
                line_dash="dashed",
                line_width=2,
            )
        )
        p_dvdt.add_layout(
            Span(
                location=normalize_window_dvdt[0],
                dimension="height",
                line_color="blue",
                line_dash="dashed",
                line_width=2,
            )
        )
        p_dvdt.add_layout(
            Span(
                location=normalize_window_dvdt[1],
                dimension="height",
                line_color="blue",
                line_dash="dashed",
                line_width=2,
            )
        )

        # Add boxzoomtool to Vm and dV/dt plots
        box_zoom_x = BoxZoomTool(dimensions="auto")
        p_vm.add_tools(box_zoom_x)
        p_vm.toolbar.active_drag = box_zoom_x
        box_zoom_x = BoxZoomTool(dimensions="auto")
        p_dvdt.add_tools(box_zoom_x)
        p_dvdt.toolbar.active_drag = box_zoom_x

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

            renderer = p_vm.multi_line(
                source=source,
                xs="xs",
                ys="ys",
                color=cluster_colors[i],
                **line_props,
                legend_label=group_label,
            )
            register_renderer(group_label, renderer)

            # Plot dV/dt traces
            df_this = df_dvdt_norm.query("ephys_roi_id in @ephys_roi_ids")
            source = ColumnDataSource(
                {
                    "xs": [df_dvdt_norm.columns.values] * len(df_this),
                    "ys": df_this.values.tolist(),
                    "ephys_roi_id": ephys_roi_ids,
                }
            )
            renderer = p_dvdt.multi_line(
                source=source,
                xs="xs",
                ys="ys",
                color=cluster_colors[i],
                **line_props,
                legend_label=group_label,
            )
            register_renderer(group_label, renderer)

            # Plot phase plot (dV/dt vs V) - normalized
            df_v_this = phase_norm_v.query("ephys_roi_id in @ephys_roi_ids")
            df_dvdt_this = phase_norm_dvdt.query("ephys_roi_id in @ephys_roi_ids")
            source = ColumnDataSource(
                {
                    "xs": df_v_this.values.tolist(),
                    "ys": df_dvdt_this.values.tolist(),
                    "ephys_roi_id": ephys_roi_ids,
                }
            )
            renderer = p_phase_norm.multi_line(
                source=source,
                xs="xs",
                ys="ys",
                color=cluster_colors[i],
                **line_props,
                legend_label=group_label,
            )
            register_renderer(group_label, renderer)

            # Plot phase plot (dV/dt vs V) - unnormalized
            if df_v_unnorm is not None and df_dvdt_unnorm is not None:
                df_v_unnorm_this = df_v_unnorm.query("ephys_roi_id in @ephys_roi_ids")
                df_dvdt_unnorm_this = df_dvdt_unnorm.query("ephys_roi_id in @ephys_roi_ids")
                source = ColumnDataSource(
                    {
                        "xs": df_v_unnorm_this.values.tolist(),
                        "ys": df_dvdt_unnorm_this.values.tolist(),
                        "ephys_roi_id": ephys_roi_ids,
                    }
                )
                renderer = p_phase.multi_line(
                    source=source,
                    xs="xs",
                    ys="ys",
                    color=cluster_colors[i],
                    **line_props,
                    legend_label=group_label,
                )
                register_renderer(group_label, renderer)

        # Add region cluster_colors to the all plots
        for region in self.df_meta["injection region"].unique():
            if region == "Non-Retro":
                continue
            roi_ids = self.df_meta.query("`injection region` == @region").ephys_roi_id.tolist()
            legend_label = f"{region}, n={len(roi_ids)}"

            source = ColumnDataSource(df_v_proj.query("ephys_roi_id in @roi_ids"))
            scatter = p_embedding.scatter(
                x=f"{dim_reduction_method}1",
                y=f"{dim_reduction_method}2",
                source=source,
                color=REGION_COLOR_MAPPER[region],
                alpha=0.8,
                size=marker_size,
                legend_label=legend_label,
            )
            scatter_renderers.append(scatter)
            register_renderer(legend_label, scatter)

            # Attach the callback to the selection changes
            source.selected.on_change("indices", partial(self.update_ephys_roi_id, source.data))

            df_v_region = df_v_norm.query("ephys_roi_id in @roi_ids")
            ys = df_v_region.values

            # Common line properties
            line_props = {
                "hover_line_color": "blue",
                "hover_line_alpha": 1.0,
                "hover_line_width": 4,
                "selection_line_color": "blue",
                "selection_line_alpha": 1.0,
                "selection_line_width": 4,
            }
            renderer = p_vm.multi_line(
                xs=[df_v_region.columns.values] * ys.shape[0],
                ys=ys.tolist(),
                color=REGION_COLOR_MAPPER[region],
                alpha=0.8,
                legend_label=legend_label,
                **line_props,
            )
            register_renderer(legend_label, renderer)
            add_timeseries_mean_sem(p_vm, df_v_region, REGION_COLOR_MAPPER[region], legend_label)

            df_dvdt_region = df_dvdt_norm.query("ephys_roi_id in @roi_ids")
            ys = df_dvdt_region.values
            renderer = p_dvdt.multi_line(
                xs=[df_dvdt_region.columns.values] * ys.shape[0],
                ys=ys.tolist(),
                color=REGION_COLOR_MAPPER[region],
                alpha=0.8,
                legend_label=legend_label,
                **line_props,
            )
            register_renderer(legend_label, renderer)
            add_timeseries_mean_sem(p_dvdt, df_dvdt_region, REGION_COLOR_MAPPER[region], legend_label)

            # Plot phase plot (dV/dt vs V) for regions - normalized
            df_v_norm_region = phase_norm_v.query("ephys_roi_id in @roi_ids")
            df_dvdt_norm_region = phase_norm_dvdt.query("ephys_roi_id in @roi_ids")
            v_vals_norm = df_v_norm_region.values
            dvdt_vals_norm = df_dvdt_norm_region.values
            renderer = p_phase_norm.multi_line(
                xs=v_vals_norm.tolist(),
                ys=dvdt_vals_norm.tolist(),
                color=REGION_COLOR_MAPPER[region],
                alpha=0.8,
                legend_label=legend_label,
                **line_props,
            )
            register_renderer(legend_label, renderer)
            add_phase_mean_sem(
                p_phase_norm,
                df_v_norm_region,
                df_dvdt_norm_region,
                REGION_COLOR_MAPPER[region],
                legend_label,
            )

            # Plot phase plot (dV/dt vs V) for regions - unnormalized
            if df_v_unnorm is not None and df_dvdt_unnorm is not None:
                v_vals_unnorm = df_v_unnorm.query("ephys_roi_id in @roi_ids").values
                dvdt_vals_unnorm = df_dvdt_unnorm.query("ephys_roi_id in @roi_ids").values
                renderer = p_phase.multi_line(
                    xs=v_vals_unnorm.tolist(),
                    ys=dvdt_vals_unnorm.tolist(),
                    color=REGION_COLOR_MAPPER[region],
                    alpha=0.8,
                    legend_label=legend_label,
                    **line_props,
                )
                register_renderer(legend_label, renderer)

        # Add tooltips
        # Add renderers like this to solve bug like this:
        #   File "/Users/han.hou/miniconda3/envs/patch-seq/lib/python3.10/
        # site-packages/panel/io/location.py", line 57, in _get_location_params
        #     params['pathname'], search = uri.split('?')
        # ValueError: too many values to unpack (expected 2)
        # 2025-04-09 00:03:04,658 500 GET /patchseq_panel_viz??? (::1) 8541.01ms
        hovertool = HoverTool(
            tooltips=self.create_tooltips(),
            renderers=scatter_renderers,
        )
        p_embedding.add_tools(hovertool)

        hovertool = HoverTool(
            tooltips=[("ephys_roi_id", "@ephys_roi_id")],
            attachment="right",  # Fix tooltip to the right of the plot
        )
        p_vm.add_tools(hovertool)
        p_dvdt.add_tools(hovertool)
        
        hovertool = HoverTool(
            tooltips=[("ephys_roi_id", "@ephys_roi_id")],
            attachment="right",
        )
        p_phase_norm.add_tools(hovertool)

        hovertool = HoverTool(
            tooltips=[("ephys_roi_id", "@ephys_roi_id")],
            attachment="right",
        )
        p_phase.add_tools(hovertool)
        
        # Add boxzoomtool to phase plot
        box_zoom_x = BoxZoomTool(dimensions="auto")
        p_phase.add_tools(box_zoom_x)
        p_phase.toolbar.active_drag = box_zoom_x

        box_zoom_x = BoxZoomTool(dimensions="auto")
        p_phase_norm.add_tools(box_zoom_x)
        p_phase_norm.toolbar.active_drag = box_zoom_x
        
        legend_configs = {
            p_vm: {"location": "top_right", "orientation": "vertical", "ncols": 1},
            p_dvdt: {"location": "top_right", "orientation": "vertical", "ncols": 1},
            p_phase: {"location": "top_left", "orientation": "vertical", "ncols": 1},
        }
        legend_font_size = max(font_size - 6, 8)

        for p in [p_embedding, p_vm, p_dvdt, p_phase_norm, p_phase]:
            if not p.legend:
                continue
            config = legend_configs.get(p)
            if config:
                p.legend.click_policy = "hide"
                for legend in p.legend:
                    legend.ncols = config.get("ncols", legend.ncols)
                    legend.background_fill_alpha = 0.5
                    legend.location = config.get("location", legend.location)
                    legend.orientation = config.get("orientation", legend.orientation)
                    legend.label_text_font_size = f"{legend_font_size}pt"
            else:
                for legend in p.legend:
                    legend.visible = False

        # Create grid layout with independent axes - now 3 rows x 2 columns
        self._sync_renderer_visibility(legend_groups)

        layout = gridplot(
            [[p_embedding, p_vm], [p_phase_norm, p_dvdt], [p_phase, None]],
            toolbar_location="right",
            merge_tools=False,
        )

        self._latest_figures = {
            "embedding": p_embedding,
            "vm": p_vm,
            "dvdt": p_dvdt,
            "phase_norm": p_phase_norm,
            "phase": p_phase,
        }

        return layout

    @staticmethod
    def _sync_renderer_visibility(legend_groups):
        """Ensure renderers with matching legends stay in sync across plots."""
        sync_code = """
        for (const target of targets) {
            if (target.visible === cb_obj.visible) {
                continue;
            }
            target.visible = cb_obj.visible;
        }
        """

        for renderers in legend_groups.values():
            if len(renderers) <= 1:
                continue
            for idx, renderer in enumerate(renderers):
                others = [r for j, r in enumerate(renderers) if j != idx]
                if not others:
                    continue
                renderer.js_on_change(
                    "visible",
                    CustomJS(args={"targets": others}, code=sync_code),
                )

    def _init_spike_subplots(
        self,
        dim_reduction_method,
        spike_range,
        normalize_window_v,
        normalize_window_dvdt,
        plot_settings,
    ):
        """Build the figures used in the spike analysis view."""
        embedding = figure(
            x_axis_label=f"{dim_reduction_method}1",
            y_axis_label=f"{dim_reduction_method}2",
            tools="pan,reset,tap,wheel_zoom,box_select,lasso_select",
            **plot_settings,
        )
        vm = figure(
            title=f"Raw Vm, normalized between {normalize_window_v[0]} to {normalize_window_v[1]} ms",
            x_axis_label="Time (ms)",
            y_axis_label="V",
            x_range=(spike_range[0] - 0.1, spike_range[1] + 0.1),
            tools="pan,reset,tap,wheel_zoom,box_select,lasso_select",
            **plot_settings,
        )
        dvdt = figure(
            title=f"dV/dt, normalized betwen {normalize_window_dvdt[0]} to {normalize_window_dvdt[1]} ms",
            x_axis_label="Time (ms)",
            y_axis_label="dV/dt",
            x_range=(spike_range[0] - 0.1, spike_range[1] + 0.1),
            tools="pan,reset,tap,wheel_zoom,box_select,lasso_select",
            **plot_settings,
        )
        phase_norm = figure(
            title="Phase Plot (Normalized)",
            x_axis_label="V (normalized)",
            y_axis_label="dV/dt (normalized)",
            tools="pan,reset,tap,wheel_zoom,box_select,lasso_select",
            **plot_settings,
        )
        phase = figure(
            title="Phase Plot (Unnormalized)",
            x_axis_label="V (mV)",
            y_axis_label="dV/dt (mV/ms)",
            tools="pan,reset,tap,wheel_zoom,box_select,lasso_select",
            **plot_settings,
        )

        return {
            "embedding": embedding,
            "vm": vm,
            "dvdt": dvdt,
            "phase_norm": phase_norm,
            "phase": phase,
        }

    @staticmethod
    def _style_subplots(figures, font_size):
        """Apply consistent font styling across subplots."""
        for fig in figures:
            fig.title.text_font_size = f"{font_size+2}pt"
            fig.xaxis.axis_label_text_font_size = f"{font_size+2}pt"
            fig.yaxis.axis_label_text_font_size = f"{font_size+2}pt"
            fig.xaxis.major_label_text_font_size = f"{font_size}pt"
            fig.yaxis.major_label_text_font_size = f"{font_size}pt"
            if fig.legend:
                fig.legend.label_text_font_size = f"{font_size}pt"


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
