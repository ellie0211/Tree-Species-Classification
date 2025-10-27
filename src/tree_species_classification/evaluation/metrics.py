"""Evaluation helpers for hierarchical models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


@dataclass
class EvaluationResult:
    leaf_report: str
    genus_report: str
    species_report: str
    leaf_f1: float
    genus_f1: float
    species_f1: float
    leaf_confusion: np.ndarray
    genus_confusion: np.ndarray
    species_confusion: np.ndarray


def evaluate_hierarchy(
    y_true_leaf: np.ndarray,
    y_pred_leaf: np.ndarray,
    y_true_genus: np.ndarray,
    y_pred_genus: np.ndarray,
    y_true_species: np.ndarray,
    y_pred_species: np.ndarray,
) -> EvaluationResult:
    return EvaluationResult(
        leaf_report=classification_report(y_true_leaf, y_pred_leaf),
        genus_report=classification_report(y_true_genus, y_pred_genus),
        species_report=classification_report(y_true_species, y_pred_species),
        leaf_f1=f1_score(y_true_leaf, y_pred_leaf, average="macro"),
        genus_f1=f1_score(y_true_genus, y_pred_genus, average="macro"),
        species_f1=f1_score(y_true_species, y_pred_species, average="macro"),
        leaf_confusion=confusion_matrix(y_true_leaf, y_pred_leaf),
        genus_confusion=confusion_matrix(y_true_genus, y_pred_genus),
        species_confusion=confusion_matrix(y_true_species, y_pred_species),
    )


__all__ = ["EvaluationResult", "evaluate_hierarchy"]
