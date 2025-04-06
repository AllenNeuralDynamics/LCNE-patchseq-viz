"""
Size mapping utilities for the scatter plot.
"""

from typing import Union

import pandas as pd
from bokeh.models import ColumnDataSource


class SizeMapping:
    """Handles size mapping for scatter plots."""

    def __init__(self, df_meta: pd.DataFrame):
        """Initialize with metadata dataframe."""
        self.df_meta = df_meta

    def determine_size_mapping(
        self,
        size_mapping: str,
        source: ColumnDataSource,
        min_size: int = 10,
        max_size: int = 20,
        gamma: float = 1,
    ) -> Union[int, str]:
        """
        Determine the size mapping for the scatter plot.

        Args:
            size_mapping: Column name to use for size mapping
            source: ColumnDataSource to add size values to
            min_size: Minimum marker size
            max_size: Maximum marker size
            gamma: Gamma value for nonlinear size scaling

        Returns:
            Either a fixed size or the name of the size column in the source
        """
        if size_mapping == "None":
            return 10

        if size_mapping in self.df_meta.columns:
            numeric_data = pd.Series(pd.to_numeric(self.df_meta[size_mapping], errors="coerce"))
            if not numeric_data.isna().all():
                # Get the min and max of the numeric data
                p5 = numeric_data.quantile(0.00)
                p95 = numeric_data.quantile(1.00)

                # Map the normalized values to sizes between min and max with
                # gamma control for nonlinearity
                normalized_values = ((numeric_data - p5) / (p95 - p5)).clip(0, 1)
                normalized_sizes = min_size + (normalized_values**gamma) * (max_size - min_size)

                # Replace NaN values with the minimum size
                normalized_sizes = normalized_sizes.fillna(5)  # Fixed size for NaN values

                # Add the size values to the source data
                source.data["size_values"] = normalized_sizes
                return "size_values"

        return 10
