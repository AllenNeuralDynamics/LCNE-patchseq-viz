"""
Spike analysis component for the visualization app.
"""

import numpy as np
import pandas as pd
import panel as pn
from bokeh.layouts import gridplot
from bokeh.models import Span, BoxZoomTool, WheelZoomTool, ColumnDataSource
from bokeh.plotting import figure
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.signal import savgol_filter

from LCNE_patchseq_analysis.pipeline_util.s3 import get_public_representative_spikes
from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
class RawSpikeAnalysis:
    """Handles spike waveform analysis and visualization."""

    def __init__(self, df_meta: pd.DataFrame):
        """Initialize with metadata dataframe."""
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
    ):
        """Extract and process representative spike waveforms."""
        # Get the waveforms
        df_waveforms = self.df_spikes.query("extract_from == @extract_from")

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
            new_dvdt[i, start_idx:start_idx + len(row)] = row

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
            new_dvdt, index=df_waveforms.index.get_level_values(0), columns=t_dvdt)

        return df_v_norm, df_dvdt_norm

    def create_plot_controls(self, width: int = 180) -> dict:
        """Create control widgets for spike analysis."""
        controls = {
            "extract_from": pn.widgets.Select(
                name="Extract spikes from",
                options=sorted(self.extract_from_options.tolist()),
                value="long_square_rheo, min",
                width=width,
            ),
            "normalize_window_v": pn.widgets.RangeSlider(
                name="V Normalization Window",
                start=-4,
                end=7,
                value=(-2, 4),
                step=0.5,
                width=width,
            ),
            "normalize_window_dvdt": pn.widgets.RangeSlider(
                name="dV/dt Normalization Window",
                start=-3,
                end=6,
                value=(-2, 0),
                step=0.5,
                width=width,
            ),
            "n_clusters": pn.widgets.IntSlider(
                name="Number of Clusters",
                start=2,
                end=7,
                value=2,
                step=1,
                width=width,
            ),
            "if_show_cluster_on_retro": pn.widgets.Checkbox(
                name="Show type color for Retro",
                value=False,
                width=width,
            ),
            "marker_size": pn.widgets.IntSlider(
                name="Marker Size",
                start=5,
                end=20,
                value=13,
                step=1,
                width=width,
            ),
            "alpha_slider": pn.widgets.FloatSlider(
                name="Alpha",
                start=0.1,
                end=1.0,
                value=0.3,
                step=0.1,
                width=width,
            ),
            "plot_width": pn.widgets.IntSlider(
                name="Plot Width",
                start=200,
                end=800,
                value=600,
                step=50,
                width=width,
            ),
            "plot_height": pn.widgets.IntSlider(
                name="Plot Height",
                start=200,
                end=800,
                value=550,
                step=50,
                width=width,
            ),
            "font_size": pn.widgets.IntSlider(
                name="Font Size",
                start=8,
                end=24,
                value=12,
                step=1,
                width=width,
            ),
        }
        return controls

    def perform_PCA_clustering(self, df_v_norm: pd.DataFrame, n_clusters: int = 2):
        """
        Perform PCA and K-means clustering on the voltage traces.
        
        Parameters:
            df_v_norm : pd.DataFrame
                Normalized voltage traces
            n_clusters : int
                Number of clusters for K-means
        
        """
        # Perform PCA
        pca = PCA()
        v = df_v_norm.values
        v_pca = pca.fit_transform(v)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(v_pca[:, :2])
        
        # Calculate metrics
        silhouette_avg = silhouette_score(v_pca[:, :2], clusters)
        metrics = {
            "silhouette_avg": silhouette_avg,
        }

        # Save data
        n_pcs = 5
        df_v_proj_PCA = pd.DataFrame(v_pca[:, :n_pcs], index=df_v_norm.index,
                                     columns=[f"raw_PC_{i}" for i in range(1, n_pcs + 1)])
        
        # Add cluster information to df_v_norm
        clusters_df = pd.DataFrame(clusters, index=df_v_norm.index, columns=["PCA_cluster_id"])
        # self.df_meta = self.df_meta.merge(clusters_df, on="ephys_roi_id", how="left")
        df_v_proj_PCA = df_v_proj_PCA.merge(clusters_df, on="ephys_roi_id", how="left")
        df_v_proj_PCA = df_v_proj_PCA.merge(self.df_meta[["ephys_roi_id", "injection region"]], 
                                            on="ephys_roi_id", 
                                            how="left")

        return df_v_proj_PCA, clusters, pca, metrics

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
    ) -> gridplot:
        """Create plots for spike analysis including PCA and clustering."""
        # Perform PCA and clustering
        df_v_proj_PCA, clusters, pca, metrics = self.perform_PCA_clustering(df_v_norm, n_clusters)        
        colors = ["black", "darkgray", "red", "green", "blue"][:n_clusters]

        # Common plot settings
        plot_settings = dict(
            width=width,
            height=height
        )

        # Create figures
        p1 = figure(
            x_axis_label="PC1",
            y_axis_label="PC2",
            tools="pan,reset,hover,tap,wheel_zoom",
            **plot_settings
        )
        p2 = figure(
            title=f"Raw Vm, normalized between {self.normalize_window_v[0]} to {self.normalize_window_v[1]} ms",
            x_axis_label="Time (ms)",
            y_axis_label="V",
            x_range=(-4.1, 7.1),
            tools="pan,reset,hover,tap",
            **plot_settings
        )
        p3 = figure(
            title=f"dV/dt, normalized betwen {self.normalize_window_dvdt[0]} to {self.normalize_window_dvdt[1]} ms",
            x_axis_label="Time (ms)",
            y_axis_label="dV/dt",
            x_range=(-3.1, 6.1),
            tools="pan,reset,hover,tap",
            **plot_settings
        )

        # Update font sizes after figure creation
        for p in [p1, p2, p3]:
            # Set the font sizes for the title and axis labels
            p.title.text_font_size = "14pt"
            p.xaxis.axis_label_text_font_size = "14pt"
            p.yaxis.axis_label_text_font_size = "14pt"

            # Set the font sizes for the major tick labels on the axes
            p.xaxis.major_label_text_font_size = "12pt"
            p.yaxis.major_label_text_font_size = "12pt"

            # Set legend font size if legend exists
            if p.legend:
                p.legend.label_text_font_size = "12pt"

        # -- Plot PCA scatter with contours --
        # Create a single ColumnDataSource for all clusters
        df_v_proj_PCA["color"] = [colors[i] for i in df_v_proj_PCA["PCA_cluster_id"]]
        # If injection region is not "Non-Retro", set color to None
        if not if_show_cluster_on_retro:
            df_v_proj_PCA.loc[df_v_proj_PCA["injection region"] != "Non-Retro", "color"] = None
        source = ColumnDataSource(df_v_proj_PCA)
        
        p1.scatter(
            x="raw_PC_1",
            y="raw_PC_2",
            source=source,
            size=marker_size,
            color='color',
            alpha=alpha,
            # legend_label=[f"Cluster {i+1}" for i in range(n_clusters)],
            hover_color="blue",
            selection_color="blue",
        )

        for i in df_v_proj_PCA["PCA_cluster_id"].unique():
            values = df_v_proj_PCA.query("PCA_cluster_id == @i").loc[:, ["raw_PC_1", "raw_PC_2"]].values
            
            # Add contours
            mean = np.mean(values, axis=0)
            cov = np.cov(values.T)
            x, y = np.mgrid[
                values[:, 0].min():values[:, 0].max():100j,
                values[:, 1].min():values[:, 1].max():100j,
            ]
            pos = np.dstack((x, y))
            rv = multivariate_normal(mean, cov)
            z = rv.pdf(pos)
            add_counter(p1, x, y, z, levels=3, line_color=colors[i], alpha=1)
            
        # Add region colors to the all plots
        for region in self.df_meta["injection region"].unique():
            if region == "Non-Retro":
                continue            
            p1.scatter(
                df_v_proj_PCA.query("`injection region` == @region").loc[:, "raw_PC_1"],
                df_v_proj_PCA.query("`injection region` == @region").loc[:, "raw_PC_2"],
                color=REGION_COLOR_MAPPER[region],
                alpha=1,
                size=marker_size,
                legend_label=region,
            )
          
        # Add metrics to the plot
        p1.title.text = f"PCA on raw + K-means Clustering (n_clusters = {n_clusters})\n" \
                        f"Silhouette Score: {metrics['silhouette_avg']:.3f}\n"
        p1.legend.click_policy = "hide"
        p1.toolbar.active_scroll = p1.select_one(WheelZoomTool)
            
        # Add vertical lines for normalization windows
        p2.add_layout(Span(
            location=self.normalize_window_v[0], dimension='height', 
            line_color='red', line_dash='dashed', line_width=2))
        p2.add_layout(Span(
            location=self.normalize_window_v[1], dimension='height', 
            line_color='red', line_dash='dashed', line_width=2))
        p3.add_layout(Span(
            location=self.normalize_window_dvdt[0], dimension='height', 
            line_color='red', line_dash='dashed', line_width=2))
        p3.add_layout(Span(
            location=self.normalize_window_dvdt[1], dimension='height', 
            line_color='red', line_dash='dashed', line_width=2))

        # Add boxzoomtool to p2 and p3
        box_zoom_x = BoxZoomTool(dimensions="auto")
        p2.add_tools(box_zoom_x)
        p2.toolbar.active_drag = box_zoom_x
        box_zoom_x = BoxZoomTool(dimensions="auto")
        p3.add_tools(box_zoom_x)
        p3.toolbar.active_drag = box_zoom_x

        # Plot voltage traces
        for i in range(n_clusters):
            mask = clusters == i
            p2.multi_line(
                [df_v_norm.columns.values] * sum(mask),
                df_v_norm.values[mask].tolist(),
                color=colors[i],
                alpha=alpha,
                hover_line_color="blue",
                hover_line_alpha=1.0,
                hover_line_width=4,
                selection_line_color="blue",
                selection_line_alpha=1.0,
                selection_line_width=4,
            )

        # Plot dV/dt traces
        for i in range(n_clusters):
            mask = clusters == i
            p3.multi_line(
                [df_dvdt_norm.columns.values] * sum(mask),
                df_dvdt_norm.values[mask].tolist(),
                color=colors[i],
                alpha=alpha,
                hover_line_color="blue",
                hover_line_alpha=1.0,
                hover_line_width=4,
                selection_line_color="blue",
                selection_line_alpha=1.0,
                selection_line_width=4,
            )
          
        
        # Create grid layout with independent axes
        layout = gridplot([[p2, p1, p3]], toolbar_location="right", merge_tools=False)
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
        level_alpha = alpha * (i/len(contour_set.allsegs))
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
        level="underlay"  # Place contour lines under other glyphs
    )
    
    # Make contour lines non-interactive
    renderer.nonselection_glyph = None  # Disable selection
    renderer.selection_glyph = None  # Disable selection
    renderer.hover_glyph = None  # Disable hover
    renderer.propagate_hover = False  # Prevent hover events from propagating
