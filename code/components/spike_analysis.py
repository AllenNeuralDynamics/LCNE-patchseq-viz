"""
Spike analysis component for the visualization app.
"""

from functools import partial

import numpy as np
import pandas as pd
import panel as pn
from bokeh.layouts import gridplot
from bokeh.models import (
    BoxZoomTool,
    CategoricalColorMapper,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    HoverTool,
    LinearColorMapper,
    Span,
    WheelZoomTool,
)
from bokeh.palettes import Blues256, Inferno256, Reds256, diverging_palette
from bokeh.plotting import figure
from scipy.stats import mannwhitneyu, multivariate_normal, ttest_ind
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:  # UMAP does not work in Hugging Face Spaces
    from umap import UMAP
except:
    UMAP = None

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.data_util.mesh import trimesh_to_bokeh_data
from LCNE_patchseq_analysis.pipeline_util.s3 import get_public_representative_spikes
from LCNE_patchseq_analysis.pipeline_util.s3 import load_mesh_from_s3


class RawSpikeAnalysis:
    """Handles spike waveform analysis and visualization."""

    def __init__(self, df_meta: pd.DataFrame, main_app):
        """Initialize with metadata dataframe."""
        self.main_app = main_app
        self.df_meta = df_meta
        self._latest_figures = {}

        # Load extracted raw spike data
        self.spike_cache = {}
        self.df_spikes = self.get_spikes("average")
        self.extract_from_options = self.df_spikes.index.get_level_values(1).unique()

    def get_spikes(self, spike_type: str) -> pd.DataFrame:
        if spike_type not in self.spike_cache:
            self.spike_cache[spike_type] = get_public_representative_spikes(spike_type)
        return self.spike_cache[spike_type]

    def create_plot_controls(self) -> dict:
        """Create control widgets for spike analysis."""
        controls = {
            "extract_from": pn.widgets.Select(
                name="Extract spikes from",
                options=sorted(self.extract_from_options.tolist()),
                value="long_square_rheo, min",
                sizing_mode="stretch_width",
            ),
            "spike_type": pn.widgets.Select(
                name="Which spike in a train",
                options=["average", "first", "second", "last"],
                value="average",
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
            "if_edge_color_projection": pn.widgets.Checkbox(
                name="Edge color by projection target",
                value=True,
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
        df_v_proj = pd.DataFrame(
            v_proj[:, :n_components], index=df_v_norm.index, columns=columns
        )

        # Add cluster information to df_v_norm
        clusters_df = pd.DataFrame(
            clusters, index=df_v_norm.index, columns=["cluster_id"]
        )
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
                    "X (A --> P)",
                    "Y (D --> V)",
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
        if_edge_color_projection: bool = True,
        spike_range: tuple = (-4, 7),
        dim_reduction_method: str = "PCA",
        normalize_window_v: tuple = (-2, 4),
        normalize_window_dvdt: tuple = (-2, 0),
    ) -> gridplot:
        """Create plots for spike analysis including dimensionality reduction and clustering."""
        # Filter data based on spike_range
        df_v_norm = df_v_norm.loc[
            :,
            (df_v_norm.columns >= spike_range[0])
            & (df_v_norm.columns <= spike_range[1]),
        ]
        df_dvdt_norm = df_dvdt_norm.loc[
            :,
            (df_dvdt_norm.columns >= spike_range[0])
            & (df_dvdt_norm.columns <= spike_range[1]),
        ]

        if df_v_phase_norm is not None:
            df_v_phase_norm = df_v_phase_norm.loc[
                :,
                (df_v_phase_norm.columns >= spike_range[0])
                & (df_v_phase_norm.columns <= spike_range[1]),
            ]
        if df_dvdt_phase_norm is not None:
            df_dvdt_phase_norm = df_dvdt_phase_norm.loc[
                :,
                (df_dvdt_phase_norm.columns >= spike_range[0])
                & (df_dvdt_phase_norm.columns <= spike_range[1]),
            ]

        # Filter unnormalized data if provided
        if df_v_unnorm is not None:
            df_v_unnorm = df_v_unnorm.loc[
                :,
                (df_v_unnorm.columns >= spike_range[0])
                & (df_v_unnorm.columns <= spike_range[1]),
            ]
        if df_dvdt_unnorm is not None:
            df_dvdt_unnorm = df_dvdt_unnorm.loc[
                :,
                (df_dvdt_unnorm.columns >= spike_range[0])
                & (df_dvdt_unnorm.columns <= spike_range[1]),
            ]

        # Perform dimensionality reduction and clustering
        df_v_proj, clusters, reducer, metrics = self.perform_dim_reduction_clustering(
            df_v_norm, n_clusters, dim_reduction_method
        )
        cluster_colors = ["black", "darkgray", "darkblue", "cyan", "darkorange"][
            :n_clusters
        ]

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

        def add_phase_mean_sem(
            fig, df_v_values, df_dvdt_values, color, label, n_bins=100
        ):
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
                    sems.append(
                        np.std(values, ddof=1) / np.sqrt(count) if count > 1 else 0.0
                    )
                if len(centers) < 2:
                    return
                centers = np.array(centers)
                means = np.array(means)
                sems = np.array(sems)
                legend_label = f"{label} (mean±SEM)"
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
            render_segment(dvdt_vals < 0, "dV/dt < 0", "solid")

        plots = self._init_spike_subplots(
            dim_reduction_method,
            spike_range,
            normalize_window_v,
            normalize_window_dvdt,
            plot_settings,
        )
        p_embedding = plots["embedding"]
        p_embedding_depth = plots["embedding_depth"]
        p_component_y = plots["component_y"]
        p_pc1_projection = plots["pc1_projection"]
        p_pc1_histogram = plots["pc1_histogram"]
        p_vm = plots["vm"]
        p_vm_depth = plots["vm_depth"]
        p_dvdt = plots["dvdt"]
        p_dvdt_depth = plots["dvdt_depth"]
        p_phase_norm = plots["phase_norm"]
        p_phase_norm_depth = plots["phase_norm_depth"]
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
            group_label = f"Cluster {i + 1}"
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
            source.selected.on_change(
                "indices", partial(self.update_ephys_roi_id, source.data)
            )

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
            add_counter(
                p_embedding, x, y, z, levels=3, line_color=cluster_colors[i], alpha=1
            )

        # Add metrics to the plot
        p_embedding.title.text = (
            f"{dim_reduction_method} + K-means Clustering (n_clusters = {n_clusters})\n"
            f"Silhouette Score: {metrics['silhouette_avg']:.3f}\n"
        )
        p_embedding.toolbar.active_scroll = p_embedding.select_one(WheelZoomTool)

        component_col = f"{dim_reduction_method}1"
        if component_col in df_v_proj.columns:
            self._add_lc_mesh_overlay(p_component_y)
            pc_values = pd.to_numeric(df_v_proj[component_col], errors="coerce")
            if pc_values.notna().any():
                # Optionally map edge color to projection target
                if if_edge_color_projection:
                    regions = df_v_proj["injection region"].unique().tolist()
                    edge_color_mapper = CategoricalColorMapper(
                        factors=regions,
                        palette=[
                            REGION_COLOR_MAPPER.get(r, "black") for r in regions
                        ],
                    )
                    line_color_spec = {
                        "field": "injection region",
                        "transform": edge_color_mapper,
                    }
                    line_width_spec = 1.5
                else:
                    line_color_spec = "black"
                    line_width_spec = 0.5

                source = ColumnDataSource(df_v_proj)
                palette = diverging_palette(Blues256, Reds256, 256)
                color_mapper = LinearColorMapper(
                    palette=palette,
                    low=float(pc_values.min()),
                    high=float(pc_values.max()),
                )
                pc_scatter = p_component_y.scatter(
                    x="X (A --> P)",
                    y="Y (D --> V)",
                    source=source,
                    size=marker_size,
                    color={"field": component_col, "transform": color_mapper},
                    line_color=line_color_spec,
                    line_width=line_width_spec,
                    alpha=0.7,
                )
                color_bar = ColorBar(color_mapper=color_mapper, width=8)
                p_component_y.add_layout(color_bar, "right")
                source.selected.on_change(
                    "indices", partial(self.update_ephys_roi_id, source.data)
                )
                hovertool = HoverTool(
                    tooltips=self.create_tooltips(),
                    renderers=[pc_scatter],
                )
                p_component_y.add_tools(hovertool)
            p_component_y.toolbar.active_scroll = p_component_y.select_one(
                WheelZoomTool
            )
            p_component_y.y_range.flipped = True

        # --- PC1 distribution by projection group (spinal cord vs cortex) ---
        spinal_regions = ["C5", "Spinal cord"]
        cortex_regions = ["Cortex", "PL", "PL, MOs"]
        pc1_spinal = pd.to_numeric(
            df_v_proj.loc[
                df_v_proj["injection region"].isin(spinal_regions), component_col
            ],
            errors="coerce",
        ).dropna()
        pc1_cortex = pd.to_numeric(
            df_v_proj.loc[
                df_v_proj["injection region"].isin(cortex_regions), component_col
            ],
            errors="coerce",
        ).dropna()

        if len(pc1_spinal) > 0 and len(pc1_cortex) > 0:
            _, mw_p = mannwhitneyu(
                pc1_spinal, pc1_cortex, alternative="two-sided"
            )
            _, tt_p = ttest_ind(pc1_spinal, pc1_cortex, equal_var=False)
            mw_str = f"p={mw_p:.2e}" if mw_p < 0.001 else f"p={mw_p:.3f}"
            tt_str = f"p={tt_p:.2e}" if tt_p < 0.001 else f"p={tt_p:.3f}"
            p_pc1_projection.title.text = (
                f"{component_col} by Projection Target\n"
                f"Mann-Whitney U: {mw_str}\nt-test: {tt_str}"
            )

            rng = np.random.default_rng(42)
            groups = [
                ("Spinal cord", pc1_spinal.values, REGION_COLOR_MAPPER["Spinal cord"]),
                ("Cortex", pc1_cortex.values, REGION_COLOR_MAPPER["Cortex"]),
            ]
            for idx, (_, data, color) in enumerate(groups):
                # Jittered strip plot
                jitter = rng.uniform(-0.15, 0.15, len(data))
                source = ColumnDataSource(
                    {"x": np.full(len(data), idx) + jitter, "y": data}
                )
                p_pc1_projection.scatter(
                    "x",
                    "y",
                    source=source,
                    size=marker_size,
                    color=color,
                    alpha=alpha,
                    line_color="black",
                    line_width=0.5,
                )

                # Box plot overlay
                q1, median, q3 = np.percentile(data, [25, 50, 75])
                iqr = q3 - q1
                upper = min(q3 + 1.5 * iqr, data.max())
                lower = max(q1 - 1.5 * iqr, data.min())

                p_pc1_projection.vbar(
                    x=[idx],
                    width=0.4,
                    top=[q3],
                    bottom=[q1],
                    fill_color=color,
                    fill_alpha=0.3,
                    line_color="black",
                    line_width=1.5,
                )
                p_pc1_projection.segment(
                    x0=[idx - 0.2], x1=[idx + 0.2],
                    y0=[median], y1=[median],
                    color="black", line_width=2.5,
                )
                # Whiskers
                p_pc1_projection.segment(
                    x0=[idx], x1=[idx], y0=[lower], y1=[q1],
                    color="black", line_width=1,
                )
                p_pc1_projection.segment(
                    x0=[idx], x1=[idx], y0=[q3], y1=[upper],
                    color="black", line_width=1,
                )
                # Whisker caps
                p_pc1_projection.segment(
                    x0=[idx - 0.1], x1=[idx + 0.1],
                    y0=[lower], y1=[lower],
                    color="black", line_width=1,
                )
                p_pc1_projection.segment(
                    x0=[idx - 0.1], x1=[idx + 0.1],
                    y0=[upper], y1=[upper],
                    color="black", line_width=1,
                )

            p_pc1_projection.xaxis.ticker = [0, 1]
            p_pc1_projection.xaxis.major_label_overrides = {
                0: f"Spinal cord\n(n={len(pc1_spinal)})",
                1: f"Cortex\n(n={len(pc1_cortex)})",
            }
            p_pc1_projection.x_range.start = -0.6
            p_pc1_projection.x_range.end = 1.6

            # Histogram of PC1 by projection group
            all_vals = np.concatenate([pc1_spinal.values, pc1_cortex.values])
            bins = np.linspace(all_vals.min(), all_vals.max(), 20)
            for grp_label, data, color in groups:
                counts, edges = np.histogram(data, bins=bins)
                source = ColumnDataSource(
                    {"top": counts, "left": edges[:-1], "right": edges[1:]}
                )
                p_pc1_histogram.quad(
                    top="top",
                    bottom=0,
                    left="left",
                    right="right",
                    source=source,
                    fill_color=color,
                    fill_alpha=0.4,
                    line_color="black",
                    line_width=0.5,
                    legend_label=f"{grp_label} (n={len(data)})",
                )
            p_pc1_histogram.legend.click_policy = "hide"
            p_pc1_histogram.title.text = (
                f"{component_col} Distribution\n"
                f"Mann-Whitney U: {mw_str}\nt-test: {tt_str}"
            )

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
            group_label = f"Cluster {i + 1}"
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
            add_timeseries_mean_sem(p_vm, df_this, cluster_colors[i], group_label)

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
            add_timeseries_mean_sem(p_dvdt, df_this, cluster_colors[i], group_label)

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
            add_phase_mean_sem(
                p_phase_norm,
                df_v_this,
                df_dvdt_this,
                cluster_colors[i],
                group_label,
            )

            # Plot phase plot (dV/dt vs V) - unnormalized
            if df_v_unnorm is not None and df_dvdt_unnorm is not None:
                df_v_unnorm_this = df_v_unnorm.query("ephys_roi_id in @ephys_roi_ids")
                df_dvdt_unnorm_this = df_dvdt_unnorm.query(
                    "ephys_roi_id in @ephys_roi_ids"
                )
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
            roi_ids = df_v_proj.query(
                "`injection region` == @region"
            ).ephys_roi_id.tolist()
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
            source.selected.on_change(
                "indices", partial(self.update_ephys_roi_id, source.data)
            )

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
            add_timeseries_mean_sem(
                p_vm, df_v_region, REGION_COLOR_MAPPER[region], legend_label
            )

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
            add_timeseries_mean_sem(
                p_dvdt, df_dvdt_region, REGION_COLOR_MAPPER[region], legend_label
            )

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
                dvdt_vals_unnorm = df_dvdt_unnorm.query(
                    "ephys_roi_id in @roi_ids"
                ).values
                renderer = p_phase.multi_line(
                    xs=v_vals_unnorm.tolist(),
                    ys=dvdt_vals_unnorm.tolist(),
                    color=REGION_COLOR_MAPPER[region],
                    alpha=0.8,
                    legend_label=legend_label,
                    **line_props,
                )
                register_renderer(legend_label, renderer)

        depth_values = pd.to_numeric(df_v_proj["Y (D --> V)"], errors="coerce")
        if depth_values.notna().any():
            depth_mapper = LinearColorMapper(
                palette=list(reversed(Inferno256)),
                low=float(depth_values.min()),
                high=float(depth_values.max()),
            )
            valid_mask = depth_values.notna()
            depth_source = ColumnDataSource(df_v_proj.loc[valid_mask])
            depth_scatter = p_embedding_depth.scatter(
                x=f"{dim_reduction_method}1",
                y=f"{dim_reduction_method}2",
                source=depth_source,
                size=marker_size,
                color={"field": "Y (D --> V)", "transform": depth_mapper},
                line_color="black",
                line_width=0.3,
                alpha=0.8,
            )
            color_bar = ColorBar(color_mapper=depth_mapper, width=8)
            p_embedding_depth.add_layout(color_bar, "right")
            depth_source.selected.on_change(
                "indices", partial(self.update_ephys_roi_id, depth_source.data)
            )
            depth_hover = HoverTool(
                tooltips=self.create_tooltips(),
                renderers=[depth_scatter],
            )
            p_embedding_depth.add_tools(depth_hover)

            missing_mask = ~valid_mask
            if missing_mask.any():
                missing_source = ColumnDataSource(df_v_proj.loc[missing_mask])
                missing_scatter = p_embedding_depth.scatter(
                    x=f"{dim_reduction_method}1",
                    y=f"{dim_reduction_method}2",
                    source=missing_source,
                    size=marker_size,
                    color="gray",
                    line_color="black",
                    line_width=0.3,
                    alpha=0.7,
                    legend_label="Depth missing",
                )
                missing_source.selected.on_change(
                    "indices", partial(self.update_ephys_roi_id, missing_source.data)
                )
                missing_hover = HoverTool(
                    tooltips=self.create_tooltips(),
                    renderers=[missing_scatter],
                )
                p_embedding_depth.add_tools(missing_hover)

            depth_map = df_v_proj.set_index("ephys_roi_id")["Y (D --> V)"]
            roi_ids = df_v_norm.index.tolist()
            depth_series = depth_map.reindex(roi_ids)

            if not depth_series.isna().all():
                depth_line_props = {
                    "hover_line_color": "blue",
                    "hover_line_alpha": 1.0,
                    "hover_line_width": 4,
                    "selection_line_color": "blue",
                    "selection_line_alpha": 1.0,
                    "selection_line_width": 4,
                }
                missing_ids = depth_series[depth_series.isna()].index.tolist()
                valid_ids = depth_series[depth_series.notna()].index.tolist()
                vm_source = ColumnDataSource(
                    {
                        "xs": [df_v_norm.columns.values] * len(valid_ids),
                        "ys": df_v_norm.loc[valid_ids].values.tolist(),
                        "depth": depth_series.loc[valid_ids].tolist(),
                        "ephys_roi_id": valid_ids,
                    }
                )
                p_vm_depth.multi_line(
                    source=vm_source,
                    xs="xs",
                    ys="ys",
                    line_color={"field": "depth", "transform": depth_mapper},
                    alpha=0.8,
                    **depth_line_props,
                )
                if missing_ids:
                    df_v_missing = df_v_norm.loc[missing_ids]
                    p_vm_depth.multi_line(
                        xs=[df_v_missing.columns.values] * len(df_v_missing),
                        ys=df_v_missing.values.tolist(),
                        line_color="gray",
                        alpha=0.6,
                        legend_label="Depth missing",
                        **depth_line_props,
                    )
                dvdt_source = ColumnDataSource(
                    {
                        "xs": [df_dvdt_norm.columns.values] * len(valid_ids),
                        "ys": df_dvdt_norm.loc[valid_ids].values.tolist(),
                        "depth": depth_series.loc[valid_ids].tolist(),
                        "ephys_roi_id": valid_ids,
                    }
                )
                p_dvdt_depth.multi_line(
                    source=dvdt_source,
                    xs="xs",
                    ys="ys",
                    line_color={"field": "depth", "transform": depth_mapper},
                    alpha=0.8,
                    **depth_line_props,
                )
                if missing_ids:
                    df_dvdt_missing = df_dvdt_norm.loc[missing_ids]
                    p_dvdt_depth.multi_line(
                        xs=[df_dvdt_missing.columns.values] * len(df_dvdt_missing),
                        ys=df_dvdt_missing.values.tolist(),
                        line_color="gray",
                        alpha=0.6,
                        legend_label="Depth missing",
                        **depth_line_props,
                    )

                phase_source = ColumnDataSource(
                    {
                        "xs": phase_norm_v.reindex(valid_ids).values.tolist(),
                        "ys": phase_norm_dvdt.reindex(valid_ids).values.tolist(),
                        "depth": depth_series.loc[valid_ids].tolist(),
                        "ephys_roi_id": valid_ids,
                    }
                )
                p_phase_norm_depth.multi_line(
                    source=phase_source,
                    xs="xs",
                    ys="ys",
                    line_color={"field": "depth", "transform": depth_mapper},
                    alpha=0.8,
                    **depth_line_props,
                )
                if missing_ids:
                    df_v_phase_missing = phase_norm_v.loc[missing_ids]
                    df_dvdt_phase_missing = phase_norm_dvdt.loc[missing_ids]
                    p_phase_norm_depth.multi_line(
                        xs=df_v_phase_missing.values.tolist(),
                        ys=df_dvdt_phase_missing.values.tolist(),
                        line_color="gray",
                        alpha=0.6,
                        legend_label="Depth missing",
                        **depth_line_props,
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
            renderers=scatter_renderers,
        )
        p_embedding.add_tools(hovertool)

        hovertool = HoverTool(
            tooltips=[("ephys_roi_id", "@ephys_roi_id")],
            attachment="right",  # Fix tooltip to the right of the plot
        )
        p_vm.add_tools(hovertool)
        p_dvdt.add_tools(hovertool)
        p_vm_depth.add_tools(hovertool)
        p_dvdt_depth.add_tools(hovertool)

        hovertool = HoverTool(
            tooltips=[("ephys_roi_id", "@ephys_roi_id")],
            attachment="right",
        )
        p_phase_norm.add_tools(hovertool)
        p_phase_norm_depth.add_tools(hovertool)

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
            p_embedding_depth: {
                "location": "top_left",
                "orientation": "vertical",
                "ncols": 1,
            },
            p_vm: {"location": "top_right", "orientation": "vertical", "ncols": 1},
            p_vm_depth: {
                "location": "top_right",
                "orientation": "vertical",
                "ncols": 1,
            },
            p_dvdt: {"location": "top_right", "orientation": "vertical", "ncols": 1},
            p_dvdt_depth: {
                "location": "top_right",
                "orientation": "vertical",
                "ncols": 1,
            },
            p_phase: {"location": "top_left", "orientation": "vertical", "ncols": 1},
            p_phase_norm_depth: {
                "location": "top_left",
                "orientation": "vertical",
                "ncols": 1,
            },
        }
        legend_font_size = max(font_size - 6, 8)

        for p in [
            p_embedding,
            p_embedding_depth,
            p_vm,
            p_vm_depth,
            p_dvdt,
            p_dvdt_depth,
            p_phase_norm,
            p_phase_norm_depth,
            p_phase,
        ]:
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
            [
                [p_embedding, p_component_y, p_pc1_projection, p_pc1_histogram],
                [p_vm, p_dvdt],
                [p_phase_norm, p_phase],
                [p_embedding_depth, p_vm_depth],
                [p_dvdt_depth, p_phase_norm_depth],
            ],
            toolbar_location="right",
            merge_tools=False,
        )

        self._latest_figures = {
            "embedding": p_embedding,
            "embedding_depth": p_embedding_depth,
            "component_y": p_component_y,
            "pc1_projection": p_pc1_projection,
            "pc1_histogram": p_pc1_histogram,
            "vm": p_vm,
            "vm_depth": p_vm_depth,
            "dvdt": p_dvdt,
            "dvdt_depth": p_dvdt_depth,
            "phase_norm": p_phase_norm,
            "phase_norm_depth": p_phase_norm_depth,
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
        embedding_depth = figure(
            title="PCA colored by depth",
            x_axis_label=f"{dim_reduction_method}1",
            y_axis_label=f"{dim_reduction_method}2",
            tools="pan,reset,tap,wheel_zoom,box_select,lasso_select",
            **plot_settings,
        )
        component_y = figure(
            title=f"{dim_reduction_method}1 in X/Y space",
            x_axis_label="X (A --> P)",
            y_axis_label="Y (D --> V)",
            tools="pan,reset,tap,wheel_zoom,box_select,lasso_select",
            match_aspect=True,
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
        vm_depth = figure(
            title="Raw Vm (depth colored)",
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
        dvdt_depth = figure(
            title="dV/dt (depth colored)",
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
        phase_norm_depth = figure(
            title="Phase Plot (depth colored)",
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
        narrow_settings = dict(
            width=max(int(plot_settings["width"] * 0.5), 300),
            height=plot_settings["height"],
        )
        pc1_projection = figure(
            title=f"{dim_reduction_method}1 by Projection Target",
            x_axis_label="Projection Target",
            y_axis_label=f"{dim_reduction_method}1",
            tools="pan,reset,wheel_zoom",
            **narrow_settings,
        )
        pc1_histogram = figure(
            title=f"{dim_reduction_method}1 Distribution",
            x_axis_label=f"{dim_reduction_method}1",
            y_axis_label="Count",
            tools="pan,reset,wheel_zoom",
            **narrow_settings,
        )

        return {
            "embedding": embedding,
            "embedding_depth": embedding_depth,
            "component_y": component_y,
            "pc1_projection": pc1_projection,
            "pc1_histogram": pc1_histogram,
            "vm": vm,
            "vm_depth": vm_depth,
            "dvdt": dvdt,
            "dvdt_depth": dvdt_depth,
            "phase_norm": phase_norm,
            "phase_norm_depth": phase_norm_depth,
            "phase": phase,
        }

    @staticmethod
    def _add_lc_mesh_overlay(fig):
        try:
            mesh = load_mesh_from_s3()
            lc_mesh_bokeh = trimesh_to_bokeh_data(mesh, direction="sagittal")
            mesh_source = ColumnDataSource(lc_mesh_bokeh)
            fig.patches(
                source=mesh_source,
                xs="xs",
                ys="ys",
                fill_alpha=0.3,
                line_color=None,
                fill_color="lightgray",
                level="underlay",
                nonselection_fill_alpha=0.3,
                nonselection_line_alpha=0,
                selection_fill_alpha=0.3,
                selection_line_alpha=0,
                muted_alpha=0.3,
            )
        except Exception:
            return

    @staticmethod
    def _style_subplots(figures, font_size):
        """Apply consistent font styling across subplots."""
        for fig in figures:
            fig.title.text_font_size = f"{font_size + 2}pt"
            fig.xaxis.axis_label_text_font_size = f"{font_size + 2}pt"
            fig.yaxis.axis_label_text_font_size = f"{font_size + 2}pt"
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
