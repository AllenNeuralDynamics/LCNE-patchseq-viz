"""Utilities for exporting Bokeh figures to SVG archives."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from typing import Dict, Any

from bokeh.io.export import get_svgs


def export_figures_to_svg_zip(figures) -> tuple[io.BytesIO, str]:
    """Return a BytesIO zip archive containing SVG exports of the provided figures.
    
    Supports both Bokeh figures and matplotlib figures wrapped in Panel panes.
    Bokeh figures are exported using Selenium WebDriver, matplotlib figures using savefig().
    
    Uses Bokeh's built-in WebDriver management for Bokeh figures. In Docker, set 
    BOKEH_CHROMEDRIVER_PATH and BOKEH_IN_DOCKER=1 environment variables for proper Chrome detection.
    
    Returns:
        tuple: (zip_buffer, timestamp) where timestamp is formatted as YYYYMMDD_HHMMSS
    """
    if not figures:
        raise ValueError("No figures available for export.")

    zip_buffer = io.BytesIO()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, fig in figures.items():
            if fig is None:
                continue
            
            # Check if it's a matplotlib figure (wrapped in Panel pane)
            if hasattr(fig, 'object') and hasattr(fig.object, 'savefig'):
                # This is a matplotlib figure wrapped in Panel pane
                matplotlib_fig = fig.object
                svg_buffer = io.BytesIO()
                matplotlib_fig.savefig(svg_buffer, format='svg', bbox_inches='tight')
                svg_buffer.seek(0)
                svg_content = svg_buffer.getvalue().decode('utf-8')
                archive.writestr(f"{timestamp}_{name}.svg", svg_content)
            else:
                # This is a Bokeh figure
                original_backend = fig.output_backend
                fig.output_backend = "svg"
                try:
                    # get_svgs() will use Bokeh's webdriver_control to manage driver lifecycle
                    svgs = get_svgs(fig)
                finally:
                    fig.output_backend = original_backend

                if not svgs:
                    continue

                svg_content = "\n".join(svgs)
                archive.writestr(f"{timestamp}_{name}.svg", svg_content)

    zip_buffer.seek(0)
    return zip_buffer, timestamp
