"""Preprocessing routines for Sentinel-2 multi-temporal stacks."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
try:  # pragma: no cover - optional dependency
    import rasterio
    from rasterio.enums import Resampling
except ImportError:  # pragma: no cover - optional dependency
    rasterio = None  # type: ignore[assignment]

    class Resampling:  # type: ignore[override]
        bilinear = "bilinear"

logger = logging.getLogger(__name__)

SCL_CLOUD_VALUES = {8, 9, 10, 11}  # see Sentinel-2 scene classification legend


@dataclass
class PreprocessingConfig:
    """Configuration for cloud masking and interpolation."""

    cloud_probability_threshold: float = 0.2
    spatial_kernel: int = 3
    temporal_window: int = 3
    resampling: Resampling = Resampling.bilinear


class Preprocessor:
    """Encapsulates preprocessing steps applied to raster time series."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        logger.debug("Preprocessor initialized with config %s", self.config)

    def run(self, files: Iterable[Path]) -> list[np.ndarray]:
        """Load, mask and interpolate Sentinel-2 mosaics."""

        stacks: list[np.ndarray] = []
        for path in files:
            logger.info("Preprocessing %s", path)
            if path.suffix == ".npz":
                content = np.load(path)
                data = content["data"]
            else:
                if rasterio is None:
                    msg = "rasterio is required to read GeoTIFF inputs"
                    raise RuntimeError(msg)
                with rasterio.open(path) as dataset:
                    data = dataset.read(out_dtype="float32", resampling=self.config.resampling)
            scl = data[-1]
            reflectance = data[:-1]
            clean = self._mask_clouds(reflectance, scl)
            interpolated = self._temporal_interpolation(clean)
            stacks.append(interpolated)
        return stacks

    def _mask_clouds(self, stack: np.ndarray, scl: np.ndarray) -> np.ndarray:
        mask = np.isin(scl, list(SCL_CLOUD_VALUES))
        masked = np.where(mask[None, ...], np.nan, stack)
        logger.debug("Applied cloud mask; %.2f%% pixels masked", float(mask.mean() * 100))
        return masked

    def _temporal_interpolation(self, stack: np.ndarray) -> np.ndarray:
        """Perform naive forward-backward fill interpolation."""

        if stack.ndim != 3:
            msg = "Stack must have shape (bands, height, width)"
            raise ValueError(msg)
        bands, height, width = stack.shape
        interpolated = stack.copy()
        for b in range(bands):
            band = interpolated[b]
            isnan = np.isnan(band)
            if np.all(isnan):
                continue
            valid_indices = np.where(~isnan)
            interpolated[b][isnan] = np.mean(band[valid_indices])
        return interpolated


__all__ = ["Preprocessor", "PreprocessingConfig"]
