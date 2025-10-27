"""Training utilities for the suite of baseline and deep models."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None  # type: ignore[assignment]
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model_name: str
    train_accuracy: float
    valid_accuracy: float
    model: object


_TorchModule = nn.Module if nn is not None else object


class TemporalCNN(_TorchModule):
    """A lightweight 1D CNN for temporal-spectral features."""

    def __init__(self, input_length: int, num_classes: int) -> None:
        if nn is None:
            msg = "PyTorch is required for CNN training"
            raise RuntimeError(msg)
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if torch is None:
            msg = "PyTorch is required for CNN inference"
            raise RuntimeError(msg)
        x = x.unsqueeze(1)
        features = self.net(x).squeeze(-1)
        return self.head(features)


class TemporalTransformer(_TorchModule):
    """Simplified Vision Transformer-style encoder for feature sequences."""

    def __init__(self, input_length: int, num_classes: int, dim: int = 128) -> None:
        if nn is None:
            msg = "PyTorch is required for ViT training"
            raise RuntimeError(msg)
        super().__init__()
        self.embed = nn.Linear(input_length, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if torch is None:
            msg = "PyTorch is required for ViT inference"
            raise RuntimeError(msg)
        x = x.unsqueeze(1)
        tokens = self.embed(x)
        encoded = self.encoder(tokens).mean(dim=1)
        return self.head(encoded)


def _train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
) -> Tuple[float, float]:
    if torch is None or nn is None or optim is None:
        msg = "PyTorch is required for deep learning models"
        raise RuntimeError(msg)
    if len(np.unique(y_train)) < 2:
        majority = int(np.bincount(y_train).argmax())
        train_acc = float(np.mean(y_train == majority))
        valid_acc = float(np.mean(y_valid == majority)) if len(y_valid) else 0.0
        return train_acc, valid_acc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).long().to(device)
    X_valid_tensor = torch.from_numpy(X_valid).float().to(device)
    y_valid_tensor = torch.from_numpy(y_valid).long().to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor).argmax(dim=1).cpu().numpy()
        valid_pred = model(X_valid_tensor).argmax(dim=1).cpu().numpy()

    return accuracy_score(y_train, train_pred), accuracy_score(y_valid, valid_pred)


def train_vit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    num_classes: int,
) -> TrainingResult:
    if len(np.unique(y_train)) < 2:
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        train_pred = dummy.predict(X_train)
        valid_pred = dummy.predict(X_valid)
        return TrainingResult("ViT", accuracy_score(y_train, train_pred), accuracy_score(y_valid, valid_pred), dummy)
    try:
        model = TemporalTransformer(X_train.shape[1], num_classes)
        train_acc, valid_acc = _train_torch_model(model, X_train, y_train, X_valid, y_valid)
        return TrainingResult("ViT", train_acc, valid_acc, model)
    except RuntimeError as exc:
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        train_pred = dummy.predict(X_train)
        valid_pred = dummy.predict(X_valid)
        logger.warning("Falling back to DummyClassifier for ViT: %s", exc)
        return TrainingResult("ViT", accuracy_score(y_train, train_pred), accuracy_score(y_valid, valid_pred), dummy)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    num_classes: int,
) -> TrainingResult:
    if len(np.unique(y_train)) < 2:
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        train_pred = dummy.predict(X_train)
        valid_pred = dummy.predict(X_valid)
        return TrainingResult("CNN", accuracy_score(y_train, train_pred), accuracy_score(y_valid, valid_pred), dummy)
    try:
        model = TemporalCNN(X_train.shape[1], num_classes)
        train_acc, valid_acc = _train_torch_model(model, X_train, y_train, X_valid, y_valid)
        return TrainingResult("CNN", train_acc, valid_acc, model)
    except RuntimeError as exc:
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        train_pred = dummy.predict(X_train)
        valid_pred = dummy.predict(X_valid)
        logger.warning("Falling back to DummyClassifier for CNN: %s", exc)
        return TrainingResult("CNN", accuracy_score(y_train, train_pred), accuracy_score(y_valid, valid_pred), dummy)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> TrainingResult:
    if len(np.unique(y_train)) < 2:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        valid_pred = model.predict(X_valid)
        return TrainingResult(
            "RandomForest", accuracy_score(y_train, train_pred), accuracy_score(y_valid, valid_pred), model
        )

    model = RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=-1)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)
    return TrainingResult(
        "RandomForest", accuracy_score(y_train, train_pred), accuracy_score(y_valid, valid_pred), model
    )


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> TrainingResult:
    if len(np.unique(y_train)) < 2:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        valid_pred = model.predict(X_valid)
        return TrainingResult("XGBoost", accuracy_score(y_train, train_pred), accuracy_score(y_valid, valid_pred), model)

    if xgb is None:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        valid_pred = model.predict(X_valid)
        return TrainingResult("XGBoost", accuracy_score(y_train, train_pred), accuracy_score(y_valid, valid_pred), model)

    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)
    return TrainingResult("XGBoost", accuracy_score(y_train, train_pred), accuracy_score(y_valid, valid_pred), model)


def train_gnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    adjacency: np.ndarray,
) -> TrainingResult:
    """Train a simple graph-aware classifier by aggregating neighbor features."""

    if adjacency.shape[0] != X_train.shape[0] + X_valid.shape[0]:
        msg = "Adjacency matrix must match total number of samples"
        raise ValueError(msg)

    features = adjacency @ np.vstack([X_train, X_valid])
    train_features = features[: X_train.shape[0]]
    valid_features = features[X_train.shape[0] :]

    if len(np.unique(y_train)) < 2:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(train_features, y_train)
        train_pred = model.predict(train_features)
        valid_pred = model.predict(valid_features)
        return TrainingResult("GNN", accuracy_score(y_train, train_pred), accuracy_score(y_valid, valid_pred), model)

    classifier = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500)
    classifier.fit(train_features, y_train)
    train_pred = classifier.predict(train_features)
    valid_pred = classifier.predict(valid_features)
    return TrainingResult("GNN", accuracy_score(y_train, train_pred), accuracy_score(y_valid, valid_pred), classifier)


__all__ = [
    "TrainingResult",
    "train_vit",
    "train_cnn",
    "train_random_forest",
    "train_xgboost",
    "train_gnn",
]
