"""Visualization helpers for species distribution maps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import folium
import geopandas as gpd
import numpy as np
from folium.plugins import HeatMap


@dataclass
class MapConfig:
    center: tuple[float, float] = (51.0, 10.0)
    zoom_start: int = 6
    output_html: Path = Path("reports/maps/species_distribution.html")


def create_species_heatmap(points: gpd.GeoDataFrame, weights: np.ndarray, config: MapConfig | None = None) -> Path:
    cfg = config or MapConfig()
    cfg.output_html.parent.mkdir(parents=True, exist_ok=True)

    m = folium.Map(location=cfg.center, zoom_start=cfg.zoom_start, tiles="CartoDB positron")

    heat_data = [
        (geom.y, geom.x, float(weight))
        for geom, weight in zip(points.geometry, weights)
        if not np.isnan(weight)
    ]
    HeatMap(heat_data, radius=12).add_to(m)

    m.save(cfg.output_html)
    return cfg.output_html


__all__ = ["MapConfig", "create_species_heatmap"]
