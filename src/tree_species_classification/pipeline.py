"""End-to-end pipeline orchestration for the tree species project."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
try:  # pragma: no cover - optional dependency
    from sentinelhub import BBox, CRS
except ImportError:  # pragma: no cover - optional dependency
    from .data.acquisition import BBox

    class CRS:  # type: ignore[override]
        WGS84 = "WGS84"

from .data.acquisition import AcquisitionConfig, SentinelDownloader
from .data.preprocessing import Preprocessor
from .evaluation.metrics import evaluate_hierarchy
from .features.extraction import FeatureExtractor
from .models.hierarchy import Hierarchy, HierarchicalClassifier
from .models.training import train_cnn, train_gnn, train_random_forest, train_vit, train_xgboost
from .utils.data import split_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class PipelineConfig:
    bbox: BBox
    start_date: str
    end_date: str
    output_dir: Path = Path("artifacts")


class ClassificationPipeline:
    def __init__(self, config: PipelineConfig, hierarchy: Hierarchy) -> None:
        self.config = config
        self.hierarchy = hierarchy

    def run(self) -> None:
        acquisition_config = AcquisitionConfig(
            bbox=self.config.bbox,
            start_date=np.datetime64(self.config.start_date).astype("datetime64[D]").astype(object),
            end_date=np.datetime64(self.config.end_date).astype("datetime64[D]").astype(object),
        )
        downloader = SentinelDownloader(acquisition_config)
        try:
            downloaded = downloader.download()
        except Exception as exc:  # pragma: no cover - network failure fallback
            logger.warning("Falling back to synthetic data due to download error: %s", exc)
            downloaded = self._create_synthetic_rasters()

        preprocessor = Preprocessor()
        stacks = preprocessor.run(downloaded)

        extractor = FeatureExtractor()
        features = extractor.transform(stacks)

        labels_leaf = np.zeros(features.shape[0], dtype=int)
        labels_genus = np.zeros_like(labels_leaf)
        labels_species = np.zeros_like(labels_leaf)

        (X_train, y_leaf_train, y_genus_train, y_species_train), (
            X_valid,
            y_leaf_valid,
            y_genus_valid,
            y_species_valid,
        ) = split_dataset(features, labels_leaf, labels_genus, labels_species)

        vit_result = train_vit(
            X_train, y_species_train, X_valid, y_species_valid, num_classes=len(self.hierarchy.species_labels)
        )
        cnn_result = train_cnn(
            X_train, y_species_train, X_valid, y_species_valid, num_classes=len(self.hierarchy.species_labels)
        )
        rf_result = train_random_forest(X_train, y_species_train, X_valid, y_species_valid)
        xgb_result = train_xgboost(X_train, y_species_train, X_valid, y_species_valid)
        adjacency = np.eye(features.shape[0])
        gnn_result = train_gnn(X_train, y_species_train, X_valid, y_species_valid, adjacency)

        best_model = max([vit_result, cnn_result, rf_result, xgb_result, gnn_result], key=lambda r: r.valid_accuracy)
        logger.info("Best species-level model: %s (valid_acc=%.3f)", best_model.model_name, best_model.valid_accuracy)

        hierarchical_classifier = HierarchicalClassifier(
            leaf_model=train_random_forest(X_train, y_leaf_train, X_valid, y_leaf_valid).model,
            genus_model=train_random_forest(X_train, y_genus_train, X_valid, y_genus_valid).model,
            species_model=best_model.model,
            hierarchy=self.hierarchy,
        )
        hierarchical_classifier.fit(X_train, y_leaf_train, y_genus_train, y_species_train)

        leaf_pred, genus_pred, species_pred = hierarchical_classifier.predict(X_valid)
        evaluation = evaluate_hierarchy(
            y_leaf_valid, leaf_pred, y_genus_valid, genus_pred, y_species_valid, species_pred
        )
        self._save_results(evaluation)

    def _save_results(self, evaluation) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_dir / "evaluation.txt", "w", encoding="utf-8") as f:
            f.write("Leaf classification report\n")
            f.write(evaluation.leaf_report)
            f.write("\nGenus classification report\n")
            f.write(evaluation.genus_report)
            f.write("\nSpecies classification report\n")
            f.write(evaluation.species_report)

    def _create_synthetic_rasters(self) -> list[Path]:
        synthetic_dir = Path("data/synthetic")
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for idx in range(5):
            path = synthetic_dir / f"synthetic_{idx}.npz"
            stack = np.random.rand(10, 32, 32).astype("float32")
            stack[-1] = np.random.randint(0, 12, size=(32, 32)).astype("float32")
            np.savez_compressed(path, data=stack)
            paths.append(path)
        return paths


def main() -> None:
    bbox = BBox(bbox=[5.5, 47.2, 15.5, 55.1], crs=CRS.WGS84)
    hierarchy = Hierarchy(leaf_labels={0: "Broadleaf", 1: "Conifer"}, genus_labels={}, species_labels={0: "Unknown"})
    pipeline = ClassificationPipeline(
        PipelineConfig(bbox=bbox, start_date="2021-04-01", end_date="2021-09-30"), hierarchy
    )
    pipeline.run()


if __name__ == "__main__":
    main()


__all__ = ["PipelineConfig", "ClassificationPipeline", "main"]
