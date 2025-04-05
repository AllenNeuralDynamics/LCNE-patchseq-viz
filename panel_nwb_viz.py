"""
Panel-based visualization tool for navigating and visualizing patch-seq NWB files.

To start the app, run:
panel serve panel_nwb_viz.py --dev --allow-websocket-origin=codeocean.allenneuraldynamics.org --title "Patch-seq Data Explorer"  # noqa: E501
"""

import logging

import pandas as pd
import panel as pn
import param
from bokeh.io import curdoc
from bokeh.layouts import column as bokeh_column
from bokeh.models import (
    BoxZoomTool,
    CategoricalColorMapper,
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
)
from bokeh.palettes import (
    Category10,
    Category20,
    Category20b,
    Category20c,
    Cividis256,
    Inferno256,
    Magma256,
    Plasma256,
    Turbo256,
    Viridis256,
)
from bokeh.plotting import figure

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.data_util.nwb import PatchSeqNWB
from LCNE_patchseq_analysis.efel.io import load_efel_features_from_roi
from LCNE_patchseq_analysis.pipeline_util.s3 import (
    get_public_url_cell_summary,
    get_public_url_sweep,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

pn.extension("tabulator")
curdoc().title = "LC-NE Patch-seq Data Explorer"

# Define available color palettes
COLOR_PALETTES = {
    "Viridis256": Viridis256,
    "Plasma256": Plasma256,
    "Magma256": Magma256,
    "Inferno256": Inferno256,
    "Cividis256": Cividis256,
    "Turbo256": Turbo256,
    "Category10": Category10,
    "Category20": Category20,
    "Category20b": Category20b,
    "Category20c": Category20c,
}


class PatchSeqNWBApp(param.Parameterized):
    """
    Object-Oriented Panel App for navigating NWB files.
    Encapsulates metadata loading, sweep visualization, and cell selection.
    """

    class DataHolder(param.Parameterized):
        """
        Holder for currently selected cell ID and sweep number.
        """
        ephys_roi_id = param.String(default="")
        sweep_number_selected = param.Integer(default=0)

    def __init__(self):
        """
        Initialize the PatchSeqNWBApp.
        """
        # Holder for currently selected cell ID.
        self.data_holder = PatchSeqNWBApp.DataHolder()

        # Load and prepare metadata.
        self.df_meta = load_ephys_metadata(if_with_efel=True)
        self.df_meta = (
            self.df_meta.rename(
                columns={col: col.replace("_tab_master", "") for col in self.df_meta.columns},
            )
            .rename(
                columns={
                    "x": "X (A --> P)",
                    "y": "Y (D --> V)",
                    "z": "Z (L --> R)",
                }
            )
            .sort_values(["injection region"])
        )

        self.df_meta["LC_targeting"].fillna("unknown", inplace=True)

        self.cell_key = [
            "Date",
            "jem-id_cell_specimen",
            "ephys_roi_id",
            "ephys_qc",
            "LC_targeting",
            "injection region",
            "Y (D --> V)",
        ]

        # Create the cell selector panel once.
        self.cell_selector_panel = self.create_cell_selector_panel()

    @staticmethod
    def update_bokeh(raw, sweep, downsample_factor=3):
        """
        Update the Bokeh plot for a given sweep.
        """
        trace = raw.get_raw_trace(sweep)[::downsample_factor]
        stimulus = raw.get_stimulus(sweep)[::downsample_factor]
        time = raw.get_time(sweep)[::downsample_factor]

        box_zoom_x = BoxZoomTool(dimensions="width")

        # Create the voltage trace plot
        voltage_plot = figure(
            title=f"Full traces - Sweep number {sweep} (downsampled {downsample_factor}x)",
            height=300,
            tools=["hover", box_zoom_x, "box_zoom", "wheel_zoom", "reset", "pan"],
            active_drag=box_zoom_x,
            x_range=(0, time[-1]),
            y_axis_label="Vm (mV)",
            sizing_mode="stretch_width",
        )
        voltage_plot.line(time, trace, line_width=1.5, color="navy")

        # Create the stimulus plot
        stim_plot = figure(
            height=150,
            tools=["hover", box_zoom_x, "box_zoom", "wheel_zoom", "reset", "pan"],
            active_drag=box_zoom_x,
            x_range=voltage_plot.x_range,  # Link x ranges
            x_axis_label="Time (ms)",
            y_axis_label="I (pA)",
            sizing_mode="stretch_width",
        )
        stim_plot.line(time, stimulus, line_width=1.5, color="firebrick")

        # Stack the plots vertically using bokeh's column layout
        layout = bokeh_column(
            voltage_plot, stim_plot, sizing_mode="stretch_width", margin=(50, 0, 0, 0)
        )
        return layout

    @staticmethod
    def highlight_selected_rows(row, highlight_subset, color, fields=None):
        """
        Highlight rows based on a subset of values.
        If fields is None, highlight the entire row.
        """
        style = [""] * len(row)
        if row["sweep_number"] in highlight_subset:
            if fields is None:
                return [f"background-color: {color}"] * len(row)
            else:
                for field in fields:
                    style[list(row.keys()).index(field)] = f"background-color: {color}"
        return style

    @staticmethod
    def get_qc_message(sweep, df_sweeps):
        """Return a QC message based on sweep data."""
        if sweep not in df_sweeps["sweep_number"].values:
            return "<span style='color:red;'>Invalid sweep!</span>"
        if sweep in df_sweeps.query("passed != passed")["sweep_number"].values:
            return "<span style='background:salmon;'>Sweep terminated by the experimenter!</span>"
        if sweep in df_sweeps.query("passed == False")["sweep_number"].values:
            return (
                f"<span style='background:yellow;'>Sweep failed QC! "
                f"({df_sweeps[df_sweeps.sweep_number == sweep].reasons.iloc[0][0]})</span>"
            )
        return "<span style='background:lightgreen;'>Sweep passed QC!</span>"

    def add_color_bar(self, color_mapper, title, p):
        """Add a color bar to the plot with consistent styling."""
        color_bar = ColorBar(
            color_mapper=color_mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
            title=title,
            title_text_font_size="12pt",
            major_label_text_font_size="10pt",
        )
        p.add_layout(color_bar, "right")
        return color_bar

    def determine_color_mapping(self, color_mapping, color_palette, p):
        """
        Determine the color mapping for the scatter plot.
        """
        if color_mapping == "injection region":
            color_mapper = {
                key: value
                for key, value in REGION_COLOR_MAPPER.items()
                if key in self.df_meta["injection region"].unique()
            }
            color_mapper = CategoricalColorMapper(
                factors=list(color_mapper.keys()), palette=list(color_mapper.values())
            )

            # Add a color bar for categorical data
            self.add_color_bar(color_mapper, color_mapping, p)
            return {"field": color_mapping, "transform": color_mapper}

        # If categorical (nunique <= 10), use categorical color mapper
        if self.df_meta[color_mapping].nunique() <= 10:
            color_mapper = CategoricalColorMapper(
                factors=list(self.df_meta[color_mapping].unique()),
                palette=color_palette[self.df_meta[color_mapping].nunique()],
            )
            self.add_color_bar(color_mapper, color_mapping, p)
            return {"field": color_mapping, "transform": color_mapper}

        # Try to convert the column to numeric
        numeric_data = pd.Series(pd.to_numeric(self.df_meta[color_mapping], errors="coerce"))
        if not numeric_data.isna().all():
            # If conversion is successful, use linear color mapper
            low = numeric_data.min()
            high = numeric_data.max()
            color_mapper = LinearColorMapper(palette=color_palette, low=low, high=high)
            color = {"field": color_mapping, "transform": color_mapper}

            # Add a color bar
            self.add_color_bar(color_mapper, color_mapping, p)
            return color

        return "black"

    def determine_size_mapping(self, size_mapping, source, min_size=10, max_size=20, gamma=1):
        """
        Determine the size mapping for the scatter plot.
        """
        if size_mapping == "None":
            return 10

        if size_mapping in self.df_meta.columns:
            numeric_data = pd.Series(pd.to_numeric(self.df_meta[size_mapping], errors="coerce"))
            if not numeric_data.isna().all():
                # Get the 5th and 95th percentiles of the numeric data
                p5 = numeric_data.quantile(0.00)
                p95 = numeric_data.quantile(1.00)

                # Map the normalized values to sizes between 10 and 20 with
                # gamma control for nonlinearity
                normalized_values = ((numeric_data - p5) / (p95 - p5)).clip(
                    0, 1
                )  # Ensure values are between 0 and 1
                normalized_sizes = min_size + (normalized_values**gamma) * (max_size - min_size)

                # Replace NaN values with the minimum size
                normalized_sizes = normalized_sizes.fillna(5)  # Fixed size for NaN values

                # Add the size values to the source data
                source.data["size_values"] = normalized_sizes
                return "size_values"

        return 10

    def update_scatter_plot(
        self,
        x_col,
        y_col,
        color_col,
        color_palette,
        size_col,
        size_range,
        size_gamma,
        alpha,
        width,
        height,
    ):
        # Create a new figure
        p = figure(
            x_axis_label=x_col,
            y_axis_label=y_col,
            tools="pan,wheel_zoom,box_zoom,reset,tap",  # ensure tap tool is included
            height=height,
            width=width,
        )

        # Create ColumnDataSource from the dataframe
        source = ColumnDataSource(self.df_meta)

        # If any column is Date, convert it to datetime
        if x_col == "Date":
            source.data[x_col] = pd.to_datetime(pd.Series(source.data[x_col]), errors="coerce")

        # Determine color mapping
        color = self.determine_color_mapping(color_col, COLOR_PALETTES[color_palette], p)

        # Determine size mapping
        size = self.determine_size_mapping(
            size_col, source, min_size=size_range[0], max_size=size_range[1], gamma=size_gamma
        )

        # Add scatter glyph using the data source
        p.scatter(x=x_col, y=y_col, source=source, size=size, color=color, alpha=alpha)

        # Flip the y-axis if y_col == "y" (depth)
        if y_col == "Y (D --> V)":
            p.y_range.flipped = True

        # Add HoverTool with tooltips
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
            # Get the actual value from the original dataframe column
            tooltips.append((f"Color ({color_col})", f"@{{{color_col}}}"))
        if size_col != "None":
            tooltips.append((f"Size ({size_col})", f"@{{{size_col}}}"))

        hovertool = HoverTool(tooltips=tooltips)
        p.add_tools(hovertool)

        # Define callback to update ephys_roi_id on point tap
        def update_ephys_roi_id(attr, old, new):
            if new:
                selected_index = new[0]
                self.data_holder.ephys_roi_id = str(
                    int(self.df_meta.iloc[selected_index]["ephys_roi_id"])
                )
                logger.info(f"Selected ephys_roi_id: {self.data_holder.ephys_roi_id}")

        # Attach the callback to the selection changes
        if hasattr(source.selected, "on_change"):
            source.selected.on_change("indices", update_ephys_roi_id)
        else:
            # Alternative method to handle selection changes
            source.selected.on("indices", update_ephys_roi_id)

        # Activate the BoxZoom tool by default
        box_zoom_tool = p.select_one(BoxZoomTool)
        if box_zoom_tool:
            p.toolbar.active_drag = box_zoom_tool

        # Set axis label font sizes
        p.xaxis.axis_label_text_font_size = "14pt"
        p.yaxis.axis_label_text_font_size = "14pt"

        # Set major tick label font sizes
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "12pt"

        return p

    def create_scatter_plot(self):
        """
        Allows the user to select any two columns from self.df_meta and
        generates a 2D scatter plot using Bokeh.
        """
        # Create dropdown widgets for selecting columns
        x_axis_select = pn.widgets.Select(
            name="X-Axis",
            options=sorted(list(self.df_meta.columns)),
            value="first_spike_AP_width @ long_square_rheo, min",
            width=200,
        )
        y_axis_select = pn.widgets.Select(
            name="Y-Axis",
            options=sorted(list(self.df_meta.columns)),
            value="Y (D --> V)",
            width=200,
        )
        color_col_select = pn.widgets.Select(
            name="Color Mapping",
            options=["None"] + sorted(list(self.df_meta.columns)),
            value="injection region",
            width=200,
        )
        size_col_select = pn.widgets.Select(
            name="Size Mapping",
            options=["None"] + sorted(list(self.df_meta.columns)),
            value="None",
            width=200,
        )
        alpha_slider = pn.widgets.FloatSlider(
            name="Alpha", value=0.7, start=0.0, end=1.0, step=0.01, width=200
        )
        # Add range slider for controlling min and max marker sizes
        size_range_slider = pn.widgets.RangeSlider(
            name="Size Range (min, max)", start=5, end=30, value=(8, 20), step=1, width=200
        )
        size_gamma_slider = pn.widgets.FloatSlider(
            name="Size Gamma", value=1, start=0.0, end=5.0, step=0.01, width=200
        )
        color_palette_select = pn.widgets.Select(
            name="Color Palette", options=list(COLOR_PALETTES.keys()), value="Viridis256", width=200
        )

        # Add plot size controls
        width_slider = pn.widgets.IntSlider(
            name="Plot Width", value=650, start=400, end=1200, step=50, width=200
        )
        height_slider = pn.widgets.IntSlider(
            name="Plot Height", value=500, start=300, end=1000, step=50, width=200
        )

        # Create a reactive scatter plot that updates when axis selections change
        scatter_plot = pn.bind(
            self.update_scatter_plot,
            x_axis_select.param.value,
            y_axis_select.param.value,
            color_col_select.param.value,
            color_palette_select.param.value,
            size_col_select.param.value,
            size_range_slider.param.value_throttled,
            size_gamma_slider.param.value_throttled,
            alpha_slider.param.value_throttled,
            width_slider.param.value_throttled,
            height_slider.param.value_throttled,
        )
        return pn.Row(
            pn.Column(
                x_axis_select,
                y_axis_select,
                color_col_select,
                size_col_select,
                pn.layout.Divider(margin=(10, 0, 10, 0)),
                pn.Accordion(
                    (
                        "Appearance Settings",
                        pn.Column(
                            color_palette_select,
                            size_range_slider,
                            size_gamma_slider,
                            alpha_slider,
                            width_slider,
                            height_slider,
                        ),
                    ),
                    active=[1],  # Open by default
                ),
                margin=(0, 20, 20, 20),
            ),
            scatter_plot,
            margin=(0, 20, 20, 20),  # top, right, bottom, left margins in pixels
        )

    def create_cell_selector_panel(self):
        """
        Builds and returns the cell selector panel that displays metadata.
        """
        # MultiSelect widget to choose additional columns.
        cols = list(self.df_meta.columns)
        cols.sort()
        selectable_cols = [col for col in cols if col not in self.cell_key]
        col_selector = pn.widgets.MultiSelect(
            name="Add Columns to show",
            options=selectable_cols,
            value=[
                "width_rheo",
                "first_spike_AP_width @ long_square_rheo, aver",
                "sag",
                "sag_ratio1 @ subthreshold, aver",
            ],  # start with no additional columns
            height=300,
            width=430,
        )

        def add_df_meta_col(selected_columns):
            return self.df_meta[self.cell_key + selected_columns]

        filtered_df_meta = pn.bind(add_df_meta_col, col_selector)
        tab_df_meta = pn.widgets.Tabulator(
            filtered_df_meta,
            selectable=1,
            disabled=True,  # Not editable
            frozen_columns=self.cell_key,
            groupby=["injection region"],
            header_filters=True,
            show_index=False,
            height=300,
            sizing_mode="stretch_width",
            pagination=None,
            stylesheets=[":host .tabulator {font-size: 12px;}"],
        )

        # When a row is selected, update the current cell (ephys_roi_id).
        def update_sweep_view_from_table(event):
            if event.new:
                selected_index = event.new[0]
                self.data_holder.ephys_roi_id = str(
                    int(self.df_meta.iloc[selected_index]["ephys_roi_id"])
                )

        tab_df_meta.param.watch(update_sweep_view_from_table, "selection")

        scatter_plot = self.create_scatter_plot()

        # Add cell-level summary plot
        def get_s3_cell_summary_plot(ephys_roi_id):
            s3_url = get_public_url_cell_summary(ephys_roi_id)
            if s3_url:
                return pn.pane.PNG(s3_url, sizing_mode="stretch_width")
            else:
                return pn.pane.Markdown("No S3 cell summary plot available")

        s3_cell_summary_plot = pn.bind(
            get_s3_cell_summary_plot, ephys_roi_id=self.data_holder.param.ephys_roi_id
        )

        cell_selector_panel = pn.Column(
            pn.Row(
                col_selector,
                tab_df_meta,
                height=350,
            ),
            pn.Row(
                scatter_plot,
                s3_cell_summary_plot,
            ),
        )
        return cell_selector_panel

    def create_sweep_panel(self, ephys_roi_id=""):
        """
        Builds and returns the sweep visualization panel for a single cell.
        """
        if ephys_roi_id == "":
            return pn.pane.Markdown("Please select a cell from the table above.")

        # Load the NWB file for the selected cell.
        raw_this_cell = PatchSeqNWB(ephys_roi_id=ephys_roi_id, if_load_metadata=False)

        # Now let's get df sweep from the eFEL enriched one
        # df_sweeps_valid = raw_this_cell.df_sweeps.query("passed == passed")
        df_sweeps = load_efel_features_from_roi(ephys_roi_id, if_from_s3=True)["df_sweeps"]
        df_sweeps_valid = df_sweeps.query("passed == passed")

        # Set initial sweep number to first valid sweep
        if self.data_holder.sweep_number_selected == 0:
            self.data_holder.sweep_number_selected = df_sweeps_valid.iloc[0]["sweep_number"]

        # Add a slider to control the downsample factor
        downsample_factor = pn.widgets.IntSlider(
            name="Downsample factor",
            value=5,
            start=1,
            end=10,
        )

        # Bind the plotting function to the data holder's sweep number
        bokeh_panel = pn.bind(
            PatchSeqNWBApp.update_bokeh,
            raw=raw_this_cell,
            sweep=self.data_holder.param.sweep_number_selected,
            downsample_factor=downsample_factor.param.value_throttled,
        )

        # Bind the S3 URL retrieval to the data holder's sweep number
        def get_s3_sweep_images(sweep_number):
            s3_url = get_public_url_sweep(ephys_roi_id, sweep_number)
            images = []
            if "sweep" in s3_url:
                images.append(pn.pane.PNG(s3_url["sweep"], width=800, height=400))
            if "spikes" in s3_url:
                images.append(pn.pane.PNG(s3_url["spikes"], width=800, height=400))
            return pn.Column(*images) if images else pn.pane.Markdown("No S3 images available")

        s3_sweep_images_panel = pn.bind(
            get_s3_sweep_images, sweep_number=self.data_holder.param.sweep_number_selected
        )
        sweep_pane = pn.Column(
            s3_sweep_images_panel,
            bokeh_panel,
            downsample_factor,
            sizing_mode="stretch_width",
        )

        # Build a Tabulator for sweep metadata.
        tab_sweeps = pn.widgets.Tabulator(
            df_sweeps_valid[
                [
                    "sweep_number",
                    "stimulus_code_ext",
                    "stimulus_name",
                    "stimulus_amplitude",
                    "passed",
                    "efel_num_spikes",
                    "num_spikes",
                    "stimulus_start_time",
                    "stimulus_duration",
                    "tags",
                    "reasons",
                    "stimulus_code",
                ]
            ],  # Only show valid sweeps (passed is not NaN)
            hidden_columns=["stimulus_code"],
            selectable=1,
            disabled=True,  # Not editable
            frozen_columns=["sweep_number"],
            header_filters=True,
            show_index=False,
            height=700,
            width=1000,
            groupby=["stimulus_code"],
            stylesheets=[":host .tabulator {font-size: 12px;}"],
        )

        # Apply conditional row highlighting.
        tab_sweeps.style.apply(
            PatchSeqNWBApp.highlight_selected_rows,
            highlight_subset=df_sweeps_valid.query("passed == True")["sweep_number"].tolist(),
            color="lightgreen",
            fields=["passed"],
            axis=1,
        ).apply(
            PatchSeqNWBApp.highlight_selected_rows,
            highlight_subset=df_sweeps_valid.query("passed != passed")["sweep_number"].tolist(),
            color="salmon",
            fields=["passed"],
            axis=1,
        ).apply(
            PatchSeqNWBApp.highlight_selected_rows,
            highlight_subset=df_sweeps_valid.query("passed == False")["sweep_number"].tolist(),
            color="yellow",
            fields=["passed"],
            axis=1,
        ).apply(
            PatchSeqNWBApp.highlight_selected_rows,
            highlight_subset=df_sweeps_valid.query("num_spikes > 0")["sweep_number"].tolist(),
            color="lightgreen",
            fields=["num_spikes"],
            axis=1,
        )

        # --- Synchronize table selection with sweep number ---
        def update_sweep_from_table(event):
            """Update sweep number when table selection changes."""
            if event.new:
                selected_index = event.new[0]
                new_sweep = df_sweeps_valid.iloc[selected_index]["sweep_number"]
                self.data_holder.sweep_number_selected = new_sweep

        tab_sweeps.param.watch(update_sweep_from_table, "selection")
        # --- End Synchronization ---

        # Build a reactive QC message panel.
        sweep_msg = pn.bind(
            PatchSeqNWBApp.get_qc_message,
            sweep=self.data_holder.param.sweep_number_selected,
            df_sweeps=df_sweeps,
        )
        sweep_msg_panel = pn.pane.Markdown(sweep_msg, width=600, height=30)

        return pn.Row(
            pn.Column(
                pn.pane.Markdown(f"# {ephys_roi_id}"),
                pn.pane.Markdown("Select a sweep from the table to view its data."),
                pn.Column(sweep_msg_panel, sweep_pane),
                width=700,
                margin=(0, 100, 0, 0),  # top, right, bottom, left margins
            ),
            pn.Column(
                pn.pane.Markdown("## Sweep metadata"),
                tab_sweeps,
            ),
        )

    def main_layout(self):
        """
        Constructs the full application layout.
        """
        pn.config.throttled = False
        pane_cell_selector = self.cell_selector_panel

        # Bind the sweep panel to the current cell selection.
        pane_one_cell = pn.bind(
            self.create_sweep_panel, ephys_roi_id=self.data_holder.param.ephys_roi_id
        )

        # Create a toggle button for showing/hiding raw sweeps
        show_sweeps_button = pn.widgets.Button(name="Show raw sweeps", button_type="primary")
        show_sweeps = pn.widgets.Toggle(name="Show raw sweeps", value=False)

        # Link the button to the toggle
        def toggle_sweeps(event):
            show_sweeps.value = not show_sweeps.value
            show_sweeps_button.name = "Hide raw sweeps" if show_sweeps.value else "Show raw sweeps"

        show_sweeps_button.on_click(toggle_sweeps)

        # Create a dynamic layout that includes pane_one_cell only when show_sweeps is True
        dynamic_content = pn.bind(
            lambda show: pn.Column(pane_one_cell) if show else pn.Column(), show_sweeps.param.value
        )

        layout = pn.Column(
            pn.pane.Markdown("# Patch-seq Ephys Data Explorer\n"),
            pn.Column(
                pn.pane.Markdown(f"## Cell selector (N = {len(self.df_meta)})"),
                width=400,
            ),
            pane_cell_selector,
            pn.layout.Divider(),
            show_sweeps_button,
            dynamic_content,
            margin=(20, 20, 0, 20),  # top, right, bottom, left margins in pixels
        )
        return layout


app = PatchSeqNWBApp()
layout = app.main_layout()
layout.servable()
