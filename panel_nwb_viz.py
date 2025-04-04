"""
Panel-based visualization tool for navigating and visualizing patch-seq NWB files.

To start the app, run:
panel serve panel_nwb_viz.py --dev --allow-websocket-origin=codeocean.allenneuraldynamics.org --title "LC-NE Patch-seq Data Explorer"  # noqa: E501
"""

import panel as pn
import param
from bokeh.io import curdoc
from bokeh.layouts import column as bokeh_column
from bokeh.models import BoxZoomTool
from bokeh.plotting import figure

from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.data_util.nwb import PatchSeqNWB
from LCNE_patchseq_analysis.pipeline_util.s3 import get_public_url_sweep

pn.extension("tabulator")
curdoc().title = "LC-NE Patch-seq Data Explorer"


class PatchSeqNWBApp(param.Parameterized):
    """
    Object-Oriented Panel App for navigating NWB files.
    Encapsulates metadata loading, sweep visualization, and cell selection.
    """

    class DataHolder(param.Parameterized):
        ephys_roi_id = param.String(default="")
        sweep_number_selected = param.Integer(default=0)

    def __init__(self):
        # Holder for currently selected cell ID.
        self.data_holder = PatchSeqNWBApp.DataHolder()

        # Load and prepare metadata.
        self.df_meta = load_ephys_metadata()
        self.df_meta = self.df_meta.rename(
            columns={col: col.replace("_tab_master", "") for col in self.df_meta.columns}
        ).sort_values(["injection region"])
        self.cell_key = [
            "Date",
            "jem-id_cell_specimen",
            "ephys_roi_id",
            "ephys_qc",
            "LC_targeting",
            "injection region",
        ]

        # Create the cell selector panel once.
        self.cell_selector_panel = self.create_cell_selector_panel()

    @staticmethod
    def update_bokeh(raw, sweep, downsample_factor=3):
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

    def create_sweep_panel(self, ephys_roi_id=""):
        """
        Builds and returns the sweep visualization panel for a single cell.
        """
        if ephys_roi_id == "":
            return pn.pane.Markdown("Please select a cell from the table above.")

        # Load the NWB file for the selected cell.
        raw_this_cell = PatchSeqNWB(ephys_roi_id=ephys_roi_id)
        df_sweeps_valid = raw_this_cell.df_sweeps.query("passed == passed")

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
        def get_s3_images(sweep_number):
            s3_url = get_public_url_sweep(ephys_roi_id, sweep_number)
            images = []
            if "sweep" in s3_url:
                images.append(pn.pane.PNG(s3_url["sweep"], width=800, height=400))
            if "spikes" in s3_url:
                images.append(pn.pane.PNG(s3_url["spikes"], width=800, height=400))
            return pn.Column(*images) if images else pn.pane.Markdown("No S3 images available")

        s3_image_panel = pn.bind(
            get_s3_images, sweep_number=self.data_holder.param.sweep_number_selected
        )
        sweep_pane = pn.Column(
            s3_image_panel,
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
            highlight_subset=raw_this_cell.df_sweeps.query("passed == True")[
                "sweep_number"
            ].tolist(),
            color="lightgreen",
            fields=["passed"],
            axis=1,
        ).apply(
            PatchSeqNWBApp.highlight_selected_rows,
            highlight_subset=raw_this_cell.df_sweeps.query("passed != passed")[
                "sweep_number"
            ].tolist(),
            color="salmon",
            fields=["passed"],
            axis=1,
        ).apply(
            PatchSeqNWBApp.highlight_selected_rows,
            highlight_subset=raw_this_cell.df_sweeps.query("passed == False")[
                "sweep_number"
            ].tolist(),
            color="yellow",
            fields=["passed"],
            axis=1,
        ).apply(
            PatchSeqNWBApp.highlight_selected_rows,
            highlight_subset=raw_this_cell.df_sweeps.query("num_spikes > 0")[
                "sweep_number"
            ].tolist(),
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
            df_sweeps=raw_this_cell.df_sweeps,
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
            value=[],  # start with no additional columns
            height=500,
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
            height=500,
            width=1300,
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

        cell_selector_panel = pn.Row(
            pn.Column(
                pn.pane.Markdown("## Cell selector"),
                pn.pane.Markdown(f"### Total LC-NE patch-seq cells: {len(self.df_meta)}"),
                width=400,
            ),
            col_selector,
            tab_df_meta,
        )
        return cell_selector_panel

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

        layout = pn.Column(
            pn.pane.Markdown("# Patch-seq Ephys Data Navigator\n"),
            pane_cell_selector,
            pn.layout.Divider(),
            pane_one_cell,
        )
        return layout


app = PatchSeqNWBApp()
layout = app.main_layout()
layout.servable()
