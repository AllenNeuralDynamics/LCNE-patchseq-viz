"""
Scatter plot component for the visualization app.
"""
import logging
from typing import List, Tuple, Any, Dict

import pandas as pd
import panel as pn
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure

from LCNE_patchseq_analysis.panel_app.components.color_mapping import ColorMapping
from LCNE_patchseq_analysis.panel_app.components.size_mapping import SizeMapping

logger = logging.getLogger(__name__)


class ScatterPlot:
    """Handles scatter plot creation and updates."""

    def __init__(self, df_meta: pd.DataFrame, data_holder: Any):
        """Initialize with metadata dataframe."""
        self.df_meta = df_meta
        self.color_mapping = ColorMapping(df_meta)
        self.size_mapping = SizeMapping(df_meta)
        self.data_holder = data_holder

    def create_plot_controls(self) -> Dict[str, Any]:
        """Create the control widgets for the scatter plot."""
        # Get numeric and categorical columns
        numeric_cols = self.df_meta.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = self.df_meta.select_dtypes(include=["object"]).columns.tolist()
        all_cols = ["None"] + sorted(numeric_cols + categorical_cols)

        controls = {
            "x_axis_select": pn.widgets.Select(
                name="X Axis",
                options=all_cols,
                value="first_spike_spike_half_width @ long_square_rheo, min",
                width=200,
            ),
            "y_axis_select": pn.widgets.Select(
                name="Y Axis",
                options=all_cols,
                value="Y (D --> V)",
                width=200,
            ),
            "color_col_select": pn.widgets.Select(
                name="Color By",
                options=all_cols,
                value="injection region",
                width=200,
            ),
            "size_col_select": pn.widgets.Select(
                name="Size By",
                options=all_cols,
                value="None",
                width=200,
            ),
            "size_range_slider": pn.widgets.RangeSlider(
                name="Size Range",
                start=5,
                end=30,
                value=(10, 20),
                step=1,
                width=200,
            ),
            "size_gamma_slider": pn.widgets.FloatSlider(
                name="Size Gamma",
                start=0.1,
                end=5,
                value=1,
                step=0.1,
                width=200,
            ),
            "alpha_slider": pn.widgets.FloatSlider(
                name="Alpha",
                start=0.1,
                end=1,
                value=0.7,
                step=0.1,
                width=200,
            ),
            "width_slider": pn.widgets.IntSlider(
                name="Width",
                start=400,
                end=1200,
                value=800,
                step=50,
                width=200,
            ),
            "height_slider": pn.widgets.IntSlider(
                name="Height",
                start=400,
                end=1200,
                value=600,
                step=50,
                width=200,
            ),
        }
        return controls

    def create_tooltips(self, x_col: str, y_col: str, color_col: str, size_col: str) -> List[Tuple[str, str]]:
        """Create tooltips for the hover tool."""
        tooltips = [
            ("Date", "@Date"),
            ("jem-id_cell_specimen", "@{jem-id_cell_specimen}"),
            ("Cell ID", "@{ephys_roi_id}"),
            ("LC_targeting", "@LC_targeting"),
            ("injection region", "@{injection region}"),
            ("---", "---"),
            ("x", f"@{{{x_col}}}"),
            ("y", f"@{{{y_col}}}"),
        ]
        
        # Add color and size mapping values to tooltips if they are selected
        if color_col != "None":
            tooltips.append(
                (f"Color ({color_col})", f"@{{{color_col}}}")
            )
        if size_col != "None":
            tooltips.append(
                (f"Size ({size_col})", f"@{{{size_col}}}")
            )
            
        return tooltips

    def update_scatter_plot(
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
    ) -> figure:
        """Update the scatter plot with new parameters."""
        # Create a new figure
        p = figure(
            x_axis_label=x_col,
            y_axis_label=y_col,
            tools="pan,wheel_zoom,box_zoom,reset,tap",
            height=height,
            width=width,
        )

        # Create ColumnDataSource from the dataframe
        source = ColumnDataSource(self.df_meta)
        
        # If any column is Date, convert it to datetime
        if x_col == "Date":
            source.data[x_col] = pd.to_datetime(pd.Series(source.data[x_col]), errors="coerce")

        # Determine color mapping
        color = self.color_mapping.determine_color_mapping(color_col, color_palette, p)
        
        # Determine size mapping
        size = self.size_mapping.determine_size_mapping(
            size_col, source, min_size=size_range[0], max_size=size_range[1], gamma=size_gamma
        )

        # Add scatter glyph using the data source
        p.scatter(x=x_col, y=y_col, source=source, size=size, color=color, alpha=alpha)

        # Flip the y-axis if y_col is depth
        if y_col == "Y (D --> V)":
            p.y_range.flipped = True

        # Add HoverTool with tooltips
        tooltips = self.create_tooltips(x_col, y_col, color_col, size_col)
        hovertool = HoverTool(tooltips=tooltips)
        p.add_tools(hovertool)

        # Define callback to update ephys_roi_id on point tap
        def update_ephys_roi_id(attr, old, new):
            if new:
                selected_index = new[0]
                ephys_roi_id = str(int(self.df_meta.iloc[selected_index]["ephys_roi_id"]))
                logger.info(f"Selected ephys_roi_id: {ephys_roi_id}")
                # Update the data holder's ephys_roi_id
                if hasattr(self, "data_holder"):
                    self.data_holder.ephys_roi_id = ephys_roi_id

        # Attach the callback to the selection changes
        source.selected.on_change("indices", update_ephys_roi_id)

        # Set axis label font sizes
        p.xaxis.axis_label_text_font_size = "14pt"
        p.yaxis.axis_label_text_font_size = "14pt"

        # Set major tick label font sizes
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "12pt"

        return p 
