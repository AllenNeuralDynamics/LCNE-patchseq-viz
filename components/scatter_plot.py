"""
Scatter plot component for the visualization app.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import panel as pn
from bokeh.layouts import gridplot
from bokeh.models import BoxZoomTool, ColumnDataSource, DatetimeTickFormatter, HoverTool, Legend
from bokeh.plotting import figure
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from LCNE_patchseq_analysis.panel_app.components.color_mapping import ColorMapping
from LCNE_patchseq_analysis.panel_app.components.size_mapping import SizeMapping
from LCNE_patchseq_analysis.pipeline_util.s3 import get_public_url_cell_summary

logger = logging.getLogger(__name__)

# Define available color palettes
COLOR_PALETTES = [
    "Viridis256",
    "Plasma256",
    "Magma256",
    "Inferno256",
    "Cividis256",
    "Turbo256",
    "Set3",
    "Category10",
    "Category20",
    "Category20b",
    "Category20c",
]


class ScatterPlot:
    """Handles scatter plot creation and updates."""

    def __init__(self, df_meta: pd.DataFrame, data_holder: Any):
        """Initialize with metadata dataframe."""
        self.df_meta = df_meta
        self.color_mapping = ColorMapping(df_meta)
        self.size_mapping = SizeMapping(df_meta)
        self.data_holder = data_holder
        # Add cell summary URLs to dataframe
        self._add_cell_summary_urls()
        self.controls = self.create_plot_controls(width=300)

    def _add_cell_summary_urls(self):
        """Add cell summary URLs to the dataframe."""
        # Create a new column for cell summary URLs
        self.df_meta["cell_summary_url"] = None

        # Get URLs for each ephys_roi_id
        for idx, row in self.df_meta.iterrows():
            ephys_roi_id = str(int(row["ephys_roi_id"]))
            try:
                url = get_public_url_cell_summary(ephys_roi_id, if_check_exists=False)
                self.df_meta.at[idx, "cell_summary_url"] = url
            except Exception as e:
                logger.warning(f"Could not get URL for ephys_roi_id {ephys_roi_id}: {e}")
                self.df_meta.at[idx, "cell_summary_url"] = None

    def create_plot_controls(self, width: int = 180) -> Dict[str, Any]:
        """Create the control widgets for the scatter plot."""
        # Get numeric and categorical columns
        numeric_cols = self.df_meta.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = self.df_meta.select_dtypes(include=["object"]).columns.tolist()
        available_cols = sorted(numeric_cols + categorical_cols)

        # Append [valid N] to the available_cols for display purposes
        available_cols = [f"{col} [valid {self.df_meta[col].count()}]" for col in available_cols]
        all_cols = ["None"] + available_cols

        controls = {
            "x_axis_select": pn.widgets.Select(
                name="X Axis",
                options=all_cols,
                value=[col for col in all_cols if "Date" in col][0],
                sizing_mode="stretch_width",
            ),
            "y_axis_select": pn.widgets.Select(
                name="Y Axis",
                options=all_cols,
                value=[
                    col
                    for col in all_cols
                    if "efel_AP_duration_half_width @ long_square_rheo, min" in col
                ][0],
                sizing_mode="stretch_width",
            ),
            "color_col_select": pn.widgets.Select(
                name="Color By",
                options=all_cols,
                value=[col for col in all_cols if "injection region" in col][0],
                sizing_mode="stretch_width",
            ),
            "color_palette_select": pn.widgets.Select(
                name="Color Palette",
                options=COLOR_PALETTES,
                value="Viridis256",
                sizing_mode="stretch_width",
            ),
            "size_col_select": pn.widgets.Select(
                name="Size By",
                options=all_cols,
                value=[col for col in all_cols if "efel_sag_ratio1 @ subthreshold, aver" in col][0],
                sizing_mode="stretch_width",
            ),
            "size_range_slider": pn.widgets.RangeSlider(
                name="Size Range",
                start=5,
                end=40,
                value=(10, 30),
                step=1,
                sizing_mode="stretch_width",
            ),
            "size_gamma_slider": pn.widgets.FloatSlider(
                name="Size Gamma",
                start=0.1,
                end=5,
                value=1,
                step=0.1,
                sizing_mode="stretch_width",
            ),
            "alpha_slider": pn.widgets.FloatSlider(
                name="Alpha",
                start=0.1,
                end=1,
                value=0.7,
                step=0.1,
                sizing_mode="stretch_width",
            ),
            "width_slider": pn.widgets.IntSlider(
                name="Width",
                start=400,
                end=1200,
                value=800,
                step=50,
                sizing_mode="stretch_width",
            ),
            "height_slider": pn.widgets.IntSlider(
                name="Height",
                start=400,
                end=1200,
                value=600,
                step=50,
                sizing_mode="stretch_width",
            ),
            "bins_slider": pn.widgets.IntSlider(
                name="Histogram bins",
                start=10,
                end=100,
                value=50,
                step=1,
                sizing_mode="stretch_width",
            ),
            "show_gmm": pn.widgets.Checkbox(
                name="Show Gaussian Mixture Model",
                value=True,
                sizing_mode="stretch_width",
            ),
            "show_linear_fit": pn.widgets.Checkbox(
                name="Show Linear Fit",
                value=True,
                sizing_mode="stretch_width",
            ),
            "n_components_x": pn.widgets.IntSlider(
                name="Number of components (X)",
                start=1,
                end=5,
                value=2,
                step=1,
                disabled=False,
                sizing_mode="stretch_width",
            ),
            "n_components_y": pn.widgets.IntSlider(
                name="Number of components (Y)",
                start=1,
                end=5,
                value=1,
                step=1,
                disabled=False,
                sizing_mode="stretch_width",
            ),
            "hist_height_slider": pn.widgets.IntSlider(
                name="Distribution plot height",
                start=50,
                end=300,
                value=150,
                step=10,
                sizing_mode="stretch_width",
            ),
            "font_size_slider": pn.widgets.IntSlider(
                name="Font Size",
                start=10,
                end=30,
                value=15,
                sizing_mode="stretch_width",
            ),
        }

        # Link the GMM checkbox to enable/disable the component sliders
        def toggle_gmm_components(event):
            controls["n_components_x"].disabled = not event.new
            controls["n_components_y"].disabled = not event.new

        controls["show_gmm"].param.watch(toggle_gmm_components, "value")

        # Initialize the disabled state based on the initial checkbox value
        controls["n_components_x"].disabled = not controls["show_gmm"].value
        controls["n_components_y"].disabled = not controls["show_gmm"].value

        return controls

    def create_tooltips(
        self, x_col: str, y_col: str, color_col: str, size_col: str
    ) -> List[Tuple[str, str]]:
        """Create tooltips for the hover tool."""

        tooltips = f"""
             <div style="text-align: left; flex: auto; white-space: nowrap; margin: 0 10px;
                       border: 2px solid black; padding: 10px;">
                    <span style="font-size: 17px;">
                        <b>@Date_str, @{{injection region}}, @{{ephys_roi_id}},
                            @{{jem-id_cell_specimen}}</b><br>
                        <b>X = @{{{x_col}}}</b> [{x_col}]<br>
                        <b>Y = @{{{y_col}}}</b> [{y_col}]<br>
                        <b> Color = @{{{color_col}}}</b> [{color_col}]<br>
                        <b> Size = @{{{size_col}}}</b> [{size_col}]<br>
                    </span>
                 <img src="@cell_summary_url{{safe}}" alt="Cell Summary"
                        style="width: 800px; height: auto;">
             </div>
             """

        return tooltips

    def create_marginal_histogram(
        self,
        data: pd.Series,
        orientation: str,
        width: int,
        height: int,
        alpha: float,
        bins: int,
        show_gmm: bool = False,
        n_components: int = 1,
    ) -> figure:
        """Create a histogram for marginal distribution with optional GMM overlay."""
        # Remove NaN values and convert to numeric
        clean_data = pd.to_numeric(data, errors="coerce").dropna()

        # If no valid data, create an empty plot
        if clean_data.empty:
            p = figure(
                height=height,
                width=width,
                tools="",
                toolbar_location=None,
                x_range=(0, 1),
                y_range=(0, 1),
            )
            p.text(
                x=0.5,
                y=0.5,
                text=["No valid data"],
                text_align="center",
                text_baseline="middle",
            )
            return p

        # Calculate histogram data (independent of orientation)
        hist, edges = np.histogram(clean_data, bins=bins, density=True)

        # Set axis ranges and quad parameters based on orientation
        if orientation == "x":
            x_range = (edges[0], edges[-1])
            y_range = (0, hist.max() * 1.1)
        else:  # "y" orientation
            x_range = (0, hist.max() * 1.1)
            y_range = (edges[0], edges[-1])

        # Create the figure
        p = figure(
            height=height,
            width=width,
            tools="",
            toolbar_location=None,
            x_range=x_range,
            y_range=y_range,
        )

        # Plot the histogram using vbar/hbar (Bokeh's bar plot) instead of quad
        if orientation == "x":
            # Use vbar for x-orientation
            p.vbar(
                x=[(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)],
                top=hist,
                width=(edges[1] - edges[0]) * 0.9,  # Slightly narrower than bin width
                fill_color="gray",
                line_color="white",
                alpha=0.9,
            )
        else:  # "y" orientation
            # Use hbar for y-orientation
            p.hbar(
                y=[(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)],
                right=hist,
                height=(edges[1] - edges[0]) * 0.9,  # Slightly narrower than bin width
                fill_color="gray",
                line_color="white",
                alpha=0.9,
            )

        # Optional: Plot Gaussian Mixture Model overlay
        if show_gmm:

            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(clean_data.values.reshape(-1, 1))
            domain = np.linspace(edges[0], edges[-1], 1000)
            density = np.exp(gmm.score_samples(domain.reshape(-1, 1)))

            # Calculate evaluation metrics
            if n_components > 1:
                labels = gmm.predict(clean_data.values.reshape(-1, 1))
                silhouette = silhouette_score(clean_data.values.reshape(-1, 1), labels)

                # Calculate BIC and AIC
                bic = gmm.bic(clean_data.values.reshape(-1, 1))
            else:
                bic = np.nan
                silhouette = np.nan

            p.line(
                *((domain, density) if orientation == "x" else (density, domain)),
                line_color="black",
                line_width=4,
                alpha=0.9,
            )

            # Plot individual components
            for i in range(n_components):
                mean = gmm.means_[i][0]
                std = np.sqrt(gmm.covariances_[i][0][0])
                weight = gmm.weights_[i]
                comp_density = (
                    weight
                    * np.exp(-0.5 * ((domain - mean) / std) ** 2)
                    / (std * np.sqrt(2 * np.pi))
                )
                p.line(
                    *((domain, comp_density) if orientation == "x" else (comp_density, domain)),
                    line_color="black",
                    line_width=2,
                    alpha=0.9,
                    line_dash="dashed",
                )

            # Add metrics to the plot title
            axis_to_show = p.xaxis if orientation == "x" else p.yaxis

            axis_to_show.axis_label = f"Silhouette: {silhouette:.3f}, " f"BIC: {bic:.3f}"

            # Font size
            axis_to_show.axis_label_text_font_size = "10pt"
            axis_to_show.major_label_text_font_size = "0pt"

        # Hide axes and grid
        axis_to_hide = p.xaxis if orientation == "y" else p.yaxis
        axis_to_hide.visible = False
        p.grid.visible = False
        return p

    def update_scatter_plot(  # noqa: C901
        self,
        x_col: str,
        y_col: str,
        color_col: str,
        color_palette: str,
        size_col: str,
        size_range: tuple,
        size_gamma: float,
        alpha: float,
        width: int,
        height: int,
        font_size: int = 14,
        bins: int = 30,
        hist_height_slider: int = 100,
        show_gmm: bool = False,
        n_components_x: int = 2,
        n_components_y: int = 1,
        show_linear_fit: bool = True,
        df_meta: pd.DataFrame = None,
    ) -> gridplot:
        """Update the scatter plot with new parameters."""
        # Use provided dataframe if supplied, otherwise use instance df_meta
        df_to_use = df_meta if df_meta is not None else self.df_meta
        
        # Strip off [valid N] from the column name
        x_col = x_col.split(" [valid ")[0]
        y_col = y_col.split(" [valid ")[0]
        color_col = color_col.split(" [valid ")[0]
        size_col = size_col.split(" [valid ")[0]

        # Create a new figure for the main scatter plot
        p = figure(
            x_axis_label=x_col,
            y_axis_label=y_col,
            tools="pan,wheel_zoom,box_zoom,reset,tap",
            height=height,
            width=width,
        )

        # Create ColumnDataSource from the dataframe
        source = ColumnDataSource(df_to_use)

        # If any column is Date, convert it to datetime
        if x_col == "Date":
            source.data[x_col] = pd.to_datetime(pd.Series(source.data[x_col]), errors="coerce")
            p.xaxis.formatter = DatetimeTickFormatter(
                years="%Y",
                months="%Y-%m",
                days="%Y-%m-%d",
            )

        # Create temporary color mapping for this specific dataframe
        temp_color_mapping = ColorMapping(df_to_use)
        # Determine color mapping
        color = temp_color_mapping.determine_color_mapping(
            color_col, color_palette, p, font_size=font_size
        )

        # Create temporary size mapping for this specific dataframe
        temp_size_mapping = SizeMapping(df_to_use)
        # Determine size mapping
        size = temp_size_mapping.determine_size_mapping(
            size_col, source, min_size=size_range[0], max_size=size_range[1], gamma=size_gamma
        )

        # Add scatter glyph using the data source
        scatter = p.scatter(x=x_col, y=y_col, source=source, size=size, color=color, alpha=alpha)

        # Add linear regression if requested and both columns are numeric
        if show_linear_fit and x_col != "Date" and x_col != "None" and y_col != "None":
            # Get clean numeric data
            # Convert to numeric and drop rows where either x or y is NA
            df_clean = df_to_use[[x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna()
            x_data = df_clean[x_col]
            y_data = df_clean[y_col]
            
            # Only proceed if we have valid data
            if not x_data.empty and not y_data.empty:
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
                
                # Calculate fitted line points
                x_min, x_max = x_data.min(), x_data.max()
                x_fit = np.array([x_min, x_max])
                y_fit = slope * x_fit + intercept
                
                # Add fitted line
                setting = {"line_width": 3, "line_dash": "solid"} if p_value < 0.05 else {
                    "line_width": 2, "line_dash": "dashed"}
                line = p.line(x_fit, y_fit, line_color="black", 
                              **setting)
                
                # Add legend with R² and p-value
                legend_items = [
                    (f"Linear Fit (p = {p_value:.3e}, R² = {r_value**2:.3f})", [line]),
                ]
                legend = Legend(
                    items=legend_items, 
                    location="top_left",
                    label_text_font_size=f"{font_size-2}pt"
                )
                p.add_layout(legend)

        # Flip the y-axis if y_col is depth
        if y_col == "Y (D --> V)":
            p.y_range.flipped = True

        # Add HoverTool with tooltips
        tooltips = self.create_tooltips(x_col, y_col, color_col, size_col)
        hovertool = HoverTool(
            tooltips=tooltips,
            attachment="right",  # Fix tooltip to the right of the plot
            formatters={"@Date": "datetime"},
        )

        p.add_tools(hovertool)

        # Define callback to update ephys_roi_id on point tap
        def update_ephys_roi_id(attr, old, new):
            if new:
                selected_index = new[0]
                ephys_roi_id = str(int(df_to_use.iloc[selected_index]["ephys_roi_id"]))
                logger.info(f"Selected ephys_roi_id: {ephys_roi_id}")
                # Update the data holder's ephys_roi_id
                if hasattr(self, "data_holder"):
                    self.data_holder.ephys_roi_id_selected = ephys_roi_id

        # Attach the callback to the selection changes
        source.selected.on_change("indices", update_ephys_roi_id)

        # Set the default tool activated on drag to be box zoom
        p.toolbar.active_drag = p.select_one(BoxZoomTool)

        # Set axis label font sizes
        p.xaxis.axis_label_text_font_size = f"{font_size}pt"
        p.yaxis.axis_label_text_font_size = f"{font_size}pt"

        # Set major tick label font sizes
        p.xaxis.major_label_text_font_size = f"{font_size*0.9}pt"
        p.yaxis.major_label_text_font_size = f"{font_size*0.9}pt"

        # Create marginal histograms
        x_hist = None
        try:
            if x_col != "Date" and x_col != "None":  # Skip histogram for Date column
                x_hist = self.create_marginal_histogram(
                    df_to_use[x_col],
                    "x",
                    width=width,
                    height=hist_height_slider,
                    alpha=alpha,
                    bins=bins,
                    show_gmm=show_gmm,
                    n_components=n_components_x,
                )
                x_hist.x_range = p.x_range  # Link x ranges
        except Exception as e:
            logger.warning(f"Could not create x histogram: {e}")
            x_hist = None

        y_hist = None
        try:
            if y_col != "Date" and y_col != "None":  # Skip histogram for Date column
                y_hist = self.create_marginal_histogram(
                    df_to_use[y_col],
                    "y",
                    width=hist_height_slider,
                    height=height,
                    alpha=alpha,
                    bins=bins,
                    show_gmm=show_gmm,
                    n_components=n_components_y,
                )
                y_hist.y_range = p.y_range  # Link y ranges
        except Exception as e:
            logger.warning(f"Could not create y histogram: {e}")
            y_hist = None

        # Count non-NaN values grouped by "injection region"
        count_non_nan = df_to_use.groupby("injection region")[[x_col, y_col]].count().T
        count_non_nan.insert(0, "Total", count_non_nan.sum(axis=1))
        count_non_nan.index = pd.Index(["X", "Y"], name="Valid N")

        # Create grid layout
        layout = pn.Column(
            gridplot(
                [[y_hist, p], [None, x_hist]],
                toolbar_location="right",
                merge_tools=True,
                toolbar_options=dict(logo=None),
            ),
            pn.pane.Markdown(count_non_nan.to_markdown()),
        )
        return layout
