"""Utility helpers for dataset splits and sampling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42


def split_dataset(
    X: np.ndarray,
    y_leaf: np.ndarray,
    y_genus: np.ndarray,
    y_species: np.ndarray,
    config: SplitConfig | None = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    cfg = config or SplitConfig()
    stratify = y_species if np.unique(y_species).size > 1 else None
    idx_train, idx_test = train_test_split(
        np.arange(X.shape[0]),
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )
    return (
        (X[idx_train], y_leaf[idx_train], y_genus[idx_train], y_species[idx_train]),
        (X[idx_test], y_leaf[idx_test], y_genus[idx_test], y_species[idx_test]),
    )


__all__ = ["SplitConfig", "split_dataset"]
