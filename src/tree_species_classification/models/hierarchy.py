"""Hierarchical classification utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


@dataclass
class Hierarchy:
    """Represents the three-level hierarchy of classes and their relationships."""

    leaf_labels: Dict[int, str]
    genus_labels: Dict[int, str]
    species_labels: Dict[int, str]
    species_to_genus: Dict[int, int] = field(default_factory=dict)
    genus_to_leaf: Dict[int, int] = field(default_factory=dict)

    def genus_for_species(self, species_id: int) -> int:
        """Return the genus identifier for a given species."""

        if self.species_to_genus:
            genus_id = self.species_to_genus.get(species_id)
            if genus_id is not None:
                return genus_id
        fallback = self._fallback_genus()
        logger.debug("Falling back to genus %s for species %s", fallback, species_id)
        return fallback

    def leaf_for_genus(self, genus_id: int) -> int:
        """Return the leaf-type identifier for a given genus."""

        if self.genus_to_leaf:
            leaf_id = self.genus_to_leaf.get(genus_id)
            if leaf_id is not None:
                return leaf_id
        fallback = self._fallback_leaf()
        logger.debug("Falling back to leaf %s for genus %s", fallback, genus_id)
        return fallback

    def leaf_for_species(self, species_id: int) -> int:
        """Convenience helper chaining ``genus_for_species`` and ``leaf_for_genus``."""

        return self.leaf_for_genus(self.genus_for_species(species_id))

    def _fallback_leaf(self) -> int:
        if not self.leaf_labels:
            return 0
        return next(iter(self.leaf_labels))

    def _fallback_genus(self) -> int:
        if not self.genus_labels:
            return 0
        return next(iter(self.genus_labels))


class HierarchicalClassifier:
    """Wraps independent classifiers for each hierarchy level."""

    def __init__(
        self,
        leaf_model: ClassifierMixin,
        genus_model: ClassifierMixin,
        species_model: ClassifierMixin,
        hierarchy: Hierarchy,
    ) -> None:
        self.leaf_model = leaf_model
        self.genus_model = genus_model
        self.species_model = species_model
        self.hierarchy = hierarchy

    def fit(self, X: np.ndarray, y_leaf: np.ndarray, y_genus: np.ndarray, y_species: np.ndarray) -> "HierarchicalClassifier":
        logger.info("Training hierarchical classifier on %s samples", X.shape[0])
        self.leaf_model.fit(X, y_leaf)
        self.genus_model.fit(X, y_genus)
        self.species_model.fit(X, y_species)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        leaf_pred = self.leaf_model.predict(X)
        genus_pred = self.genus_model.predict(X)
        species_pred = self.species_model.predict(X)
        return leaf_pred, genus_pred, species_pred

    def score(self, X: np.ndarray, y_leaf: np.ndarray, y_genus: np.ndarray, y_species: np.ndarray) -> Dict[str, float]:
        leaf_pred, genus_pred, species_pred = self.predict(X)
        return {
            "leaf_accuracy": accuracy_score(y_leaf, leaf_pred),
            "genus_accuracy": accuracy_score(y_genus, genus_pred),
            "species_accuracy": accuracy_score(y_species, species_pred),
        }


__all__ = ["Hierarchy", "HierarchicalClassifier"]
