"""
Color mapping utilities for the scatter plot.
"""

from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, ColorBar, LinearColorMapper
from bokeh.palettes import all_palettes
from bokeh.plotting import figure

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER


class ColorMapping:
    """Handles color mapping for scatter plots."""

    def __init__(self, df_meta: pd.DataFrame, font_size: int = 14):
        """Initialize with metadata dataframe."""
        self.df_meta = df_meta
        self.font_size = font_size

    def add_color_bar(
        self,
        color_mapper: Union[CategoricalColorMapper, LinearColorMapper],
        title: str,
        p: figure,
        font_size: int = 14,
    ) -> ColorBar:
        """Add a color bar to the plot with consistent styling."""
        color_bar = ColorBar(
            color_mapper=color_mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
            title=title,
            title_text_font_size=f"{font_size*0.8}pt",
            major_label_text_font_size=f"{font_size*0.8}pt",
        )
        p.add_layout(color_bar, "right")
        return color_bar

    def determine_color_mapping(  # noqa: C901
        self, color_mapping: str, color_palette: Any, p: figure, font_size: int = 14
    ) -> Dict[str, Any]:
        """
        Determine the color mapping for the scatter plot.

        Args:
            color_mapping: Column name to use for color mapping
            color_palette: Color palette to use
            p: Bokeh figure to add color bar to

        Returns:
            Dictionary with field and transform for scatter plot
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
            self.add_color_bar(color_mapper, color_mapping, p, font_size)

            return {"field": color_mapping, "transform": color_mapper}

        # If categorical (nunique <= 50), use categorical color mapper
        if self.df_meta[color_mapping].nunique() <= 50:
            n_categories = self.df_meta[color_mapping].nunique()

            # Check if the provided color_palette is a string (name in all_palettes)
            if isinstance(color_palette, str) and color_palette in all_palettes:
                # Use the named palette from all_palettes
                # Check if the number of categories is supported by the palette
                max_colors = max(all_palettes[color_palette].keys())
                n_colors = min(n_categories, max_colors)
                categorical_palette = all_palettes[color_palette][n_colors]
                # Make the palette circular by cycling through the colors
                if n_categories > n_colors:
                    categorical_palette = list(categorical_palette)
                    categorical_palette = [
                        categorical_palette[i % n_colors] for i in range(n_categories)
                    ]
            else:
                # For continuous palette lists or palette names not in all_palettes
                # First check if it's a named continuous palette like 'Viridis256'
                named_palette = None
                for name, palettes in all_palettes.items():
                    if isinstance(color_palette, str) and color_palette.startswith(name):
                        # If we found the palette, use it
                        if 256 in palettes:
                            named_palette = palettes[256]
                        elif len(palettes) > 0:
                            # Get the largest available palette
                            max_key = max(palettes.keys())
                            named_palette = palettes[max_key]
                        break

                # Use the named continuous palette or the provided palette
                if named_palette is not None:
                    continuous_palette = named_palette
                else:
                    continuous_palette = color_palette

                # Uniformly sample from the continuous colormap
                if isinstance(continuous_palette, (list, tuple)):
                    indices = np.linspace(0, len(continuous_palette) - 1, n_categories).astype(int)
                    categorical_palette = [continuous_palette[i] for i in indices]
                else:
                    # If we can't determine the palette type, use a default
                    categorical_palette = all_palettes["Category10"][min(n_categories, 10)]

            # Map "None" or NaN factors to "gray"
            factors = list(self.df_meta[color_mapping].dropna().unique())
            categorical_palette = list(categorical_palette)
            for missing in ["None", "unknown", "seq_data_not_available"]:
                if missing in factors:
                    categorical_palette[factors.index(missing)] = "gray"

            color_mapper = CategoricalColorMapper(
                factors=factors,
                palette=categorical_palette,
            )
            self.add_color_bar(color_mapper, color_mapping, p)
            return {"field": color_mapping, "transform": color_mapper}

        # Try to convert the column to numeric
        numeric_data = pd.Series(pd.to_numeric(self.df_meta[color_mapping], errors="coerce"))
        if not numeric_data.isna().all():
            # If conversion is successful, use linear color mapper
            low = numeric_data.quantile(0.01)
            high = numeric_data.quantile(0.99)
            color_mapper = LinearColorMapper(palette=color_palette, low=low, high=high)
            color = {"field": color_mapping, "transform": color_mapper}

            # Add a color bar
            self.add_color_bar(color_mapper, color_mapping, p)
            return color

        return "black"
