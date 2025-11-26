"""Utilities for exporting Bokeh figures to SVG archives."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from typing import Dict

from bokeh.io.export import get_svgs


def export_figures_to_svg_zip(figures) -> tuple[io.BytesIO, str]:
    """Return a BytesIO zip archive containing SVG exports of the provided figures.
    
    Uses Bokeh's built-in WebDriver management. In Docker, set BOKEH_CHROMEDRIVER_PATH
    and BOKEH_IN_DOCKER=1 environment variables for proper Chrome detection.
    
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
