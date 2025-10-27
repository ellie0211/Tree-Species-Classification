"""Feature engineering utilities for multi-temporal Sentinel-2 stacks."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration flags for feature extraction."""

    include_indices: bool = True
    include_statistics: bool = True


class FeatureExtractor:
    """Transforms preprocessed Sentinel-2 stacks into feature vectors."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def transform(self, stacks: Iterable[np.ndarray]) -> np.ndarray:
        features = [self._stack_to_features(stack) for stack in stacks]
        return np.vstack(features)

    def _stack_to_features(self, stack: np.ndarray) -> np.ndarray:
        if stack.ndim != 3:
            msg = "Stack must be 3-dimensional (bands, height, width)"
            raise ValueError(msg)

        flat_pixels = stack.reshape(stack.shape[0], -1)
        feature_parts: list[np.ndarray] = [flat_pixels.mean(axis=1), flat_pixels.std(axis=1)]

        if self.config.include_indices:
            feature_parts.append(self._vegetation_indices(flat_pixels))

        if self.config.include_statistics:
            feature_parts.append(self._temporal_statistics(flat_pixels))

        feature_vector = np.concatenate(feature_parts)
        logger.debug("Extracted feature vector of shape %s", feature_vector.shape)
        return feature_vector

    def _vegetation_indices(self, pixels: np.ndarray) -> np.ndarray:
        b03, b04, b08, b11 = pixels[1], pixels[2], pixels[5], pixels[7]
        ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
        evi = 2.5 * (b08 - b04) / (b08 + 6 * b04 - 7.5 * b03 + 1)
        savi = (1.5 * (b08 - b04)) / (b08 + b04 + 0.5)
        ndwi = (b08 - b11) / (b08 + b11 + 1e-6)
        msi = b11 / (b08 + 1e-6)
        return np.array([ndvi.mean(), evi.mean(), savi.mean(), ndwi.mean(), msi.mean()])

    def _temporal_statistics(self, pixels: np.ndarray) -> np.ndarray:
        return np.concatenate([pixels.min(axis=1), pixels.max(axis=1)])


__all__ = ["FeatureConfig", "FeatureExtractor"]
