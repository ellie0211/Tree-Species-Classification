"""Data acquisition utilities for Sentinel-2 imagery.

This module provides abstractions to download and cache multi-temporal
Sentinel-2 Level-2A imagery for the tree species classification project.
The goal is to keep the logic testable and mockable by avoiding any
side-effects in the public API.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:  # pragma: no cover - optional dependency
    from sentinelhub import BBox, DataCollection, MimeType, SHConfig, SentinelHubRequest
except ImportError as exc:  # pragma: no cover - optional dependency
    class BBox:  # type: ignore[override]
        """Minimal placeholder used when Sentinel Hub SDK is unavailable."""

        def __init__(self, bbox: list[float], crs: object | None = None) -> None:
            self.min_x, self.min_y, self.max_x, self.max_y = bbox

    class DataCollection:  # type: ignore[override]
        SENTINEL2_L2A = "SENTINEL2_L2A"

    class MimeType:  # type: ignore[override]
        TIFF = "TIFF"

    class SHConfig:  # type: ignore[override]
        pass

    class SentinelHubRequest:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Sentinel Hub SDK is required for data acquisition") from exc


logger = logging.getLogger(__name__)


@dataclass
class AcquisitionConfig:
    """Configuration parameters that describe a Sentinel-2 download job."""

    bbox: BBox
    start_date: dt.date
    end_date: dt.date
    mosaicking_order: str = "leastCC"
    max_cloud_cover: float = 0.2
    time_interval_days: int = 5
    data_folder: Path = Path("data/raw")
    config: Optional[SHConfig] = None

    def __post_init__(self) -> None:
        if self.end_date < self.start_date:
            msg = "end_date must be after start_date"
            raise ValueError(msg)
        if not 0 <= self.max_cloud_cover <= 1:
            msg = "max_cloud_cover must be in the range [0, 1]"
            raise ValueError(msg)
        self.data_folder.mkdir(parents=True, exist_ok=True)


class SentinelDownloader:
    """Handles retrieval of Sentinel-2 data via the SentinelHub API."""

    def __init__(self, acquisition_config: AcquisitionConfig) -> None:
        self.config = acquisition_config
        self._sh_config = acquisition_config.config or SHConfig()
        logger.debug("Initialized SentinelDownloader with config %s", acquisition_config)

    def _build_request(self, time_interval: tuple[str, str]) -> SentinelHubRequest:
        """Create the Sentinel Hub request for the provided time interval."""

        logger.debug("Building request for time interval %s", time_interval)
        return SentinelHubRequest(
            data_folder=str(self.config.data_folder),
            evalscript=self._evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    mosaicking_order=self.config.mosaicking_order,
                    maxcc=self.config.max_cloud_cover,
                    time_interval=time_interval,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=self.config.bbox,
            size=self._tile_size,
            config=self._sh_config,
        )

    @property
    def _tile_size(self) -> tuple[int, int]:
        """Compute the tile size in pixels for a 10m resolution scene."""

        width = int((self.config.bbox.max_x - self.config.bbox.min_x) / 10)
        height = int((self.config.bbox.max_y - self.config.bbox.min_y) / 10)
        logger.debug("Computed tile size (width=%s, height=%s)", width, height)
        return width, height

    @property
    def _evalscript(self) -> str:
        """Evalscript requesting Level-2A bands and SCL mask."""

        return """
            //VERSION=3
            function setup() {
              return {
                input: [{
                  bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "SCL"],
                  units: "REFLECTANCE"
                }],
                output: {
                  bands: 10,
                  sampleType: "FLOAT32"
                }
              };
            }

            function evaluatePixel(sample) {
              return [sample.B02, sample.B03, sample.B04, sample.B05, sample.B06,
                      sample.B07, sample.B08, sample.B11, sample.B12, sample.SCL];
            }
        """

    def download(self) -> list[Path]:
        """Download the data across the configured date range."""

        dates = _generate_time_slices(
            self.config.start_date, self.config.end_date, self.config.time_interval_days
        )
        downloaded_files: list[Path] = []

        for start, end in dates:
            request = self._build_request((start.isoformat(), end.isoformat()))
            logger.info("Triggering Sentinel Hub request for %s to %s", start, end)
            data_path = Path(request.save_data())
            logger.debug("Downloaded data saved to %s", data_path)
            downloaded_files.extend(data_path.glob("*.tif"))

        return downloaded_files


def _generate_time_slices(
    start_date: dt.date, end_date: dt.date, interval_days: int
) -> Iterable[tuple[dt.date, dt.date]]:
    """Yield start/end date tuples covering the requested time range."""

    current = start_date
    while current <= end_date:
        next_date = min(current + dt.timedelta(days=interval_days - 1), end_date)
        yield current, next_date
        current = next_date + dt.timedelta(days=1)


__all__ = ["AcquisitionConfig", "SentinelDownloader"]
