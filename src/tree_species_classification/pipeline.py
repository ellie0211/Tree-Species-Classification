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
    apply_smote: bool = False
    random_seed: int = 42


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
        using_synthetic = False
        try:
            downloaded = downloader.download()
        except Exception as exc:  # pragma: no cover - network failure fallback
            logger.warning("Falling back to synthetic data due to download error: %s", exc)
            downloaded = self._create_synthetic_rasters()
            using_synthetic = True

        preprocessor = Preprocessor()
        stacks = preprocessor.run(downloaded)

        extractor = FeatureExtractor()
        features = extractor.transform(stacks)

        labels_leaf, labels_genus, labels_species = self._generate_hierarchical_labels(
            features.shape[0], synthetic=using_synthetic
        )

        (X_train, y_leaf_train, y_genus_train, y_species_train), (
            X_valid,
            y_leaf_valid,
            y_genus_valid,
            y_species_valid,
        ) = split_dataset(features, labels_leaf, labels_genus, labels_species)

        if self.config.apply_smote:
            X_train, y_leaf_train, y_genus_train, y_species_train = self._apply_smote(
                X_train, y_leaf_train, y_genus_train, y_species_train
            )

        vit_result = train_vit(
            X_train, y_species_train, X_valid, y_species_valid, num_classes=len(self.hierarchy.species_labels)
        )
        cnn_result = train_cnn(
            X_train, y_species_train, X_valid, y_species_valid, num_classes=len(self.hierarchy.species_labels)
        )
        rf_result = train_random_forest(X_train, y_species_train, X_valid, y_species_valid)
        xgb_result = train_xgboost(X_train, y_species_train, X_valid, y_species_valid)
        adjacency = np.eye(X_train.shape[0] + X_valid.shape[0])
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
            f.write("\n\nMacro F1 scores (leaf/genus/species): ")
            f.write(
                f"{evaluation.leaf_f1:.3f} / {evaluation.genus_f1:.3f} / {evaluation.species_f1:.3f}"
            )

        np.savetxt(self.config.output_dir / "leaf_confusion.csv", evaluation.leaf_confusion, fmt="%d", delimiter=",")
        np.savetxt(
            self.config.output_dir / "genus_confusion.csv",
            evaluation.genus_confusion,
            fmt="%d",
            delimiter=",",
        )
        np.savetxt(
            self.config.output_dir / "species_confusion.csv",
            evaluation.species_confusion,
            fmt="%d",
            delimiter=",",
        )

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

    def _generate_hierarchical_labels(self, num_samples: int, synthetic: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if num_samples == 0:
            return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

        rng = np.random.default_rng(self.config.random_seed)
        species_ids = np.array(sorted(self.hierarchy.species_labels))
        if len(species_ids) == 0:
            zeros = np.zeros(num_samples, dtype=int)
            return zeros, zeros.copy(), zeros.copy()

        if synthetic:
            weights = np.geomspace(1.0, 0.2, num=len(species_ids))
            probabilities = weights / weights.sum()
        else:
            probabilities = None

        y_species = rng.choice(species_ids, size=num_samples, p=probabilities)
        y_genus = np.array([self.hierarchy.genus_for_species(s) for s in y_species], dtype=int)
        y_leaf = np.array([self.hierarchy.leaf_for_genus(g) for g in y_genus], dtype=int)
        return y_leaf, y_genus, y_species

    def _apply_smote(
        self,
        X: np.ndarray,
        y_leaf: np.ndarray,
        y_genus: np.ndarray,
        y_species: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:  # pragma: no cover - optional dependency
            logger.warning("SMOTE requested but imbalanced-learn is not installed; skipping oversampling")
            return X, y_leaf, y_genus, y_species

        if len(np.unique(y_species)) < 2:
            logger.info("Skipping SMOTE because only one species class is present")
            return X, y_leaf, y_genus, y_species

        smote = SMOTE(random_state=self.config.random_seed)
        X_resampled, y_species_resampled = smote.fit_resample(X, y_species)
        y_genus_resampled = np.array(
            [self.hierarchy.genus_for_species(species) for species in y_species_resampled], dtype=y_genus.dtype
        )
        y_leaf_resampled = np.array(
            [self.hierarchy.leaf_for_genus(genus) for genus in y_genus_resampled], dtype=y_leaf.dtype
        )
        logger.info(
            "Applied SMOTE: %s -> %s samples", X.shape[0], X_resampled.shape[0]
        )
        return X_resampled, y_leaf_resampled, y_genus_resampled, y_species_resampled


def main() -> None:
    bbox = BBox(bbox=[5.5, 47.2, 15.5, 55.1], crs=CRS.WGS84)
    hierarchy = Hierarchy(
        leaf_labels={0: "Broadleaf", 1: "Conifer"},
        genus_labels={0: "Fagus", 1: "Picea", 2: "Pinus"},
        species_labels={0: "Fagus sylvatica", 1: "Picea abies", 2: "Pinus sylvestris"},
        species_to_genus={0: 0, 1: 1, 2: 2},
        genus_to_leaf={0: 0, 1: 1, 2: 1},
    )
    pipeline = ClassificationPipeline(
        PipelineConfig(bbox=bbox, start_date="2021-04-01", end_date="2021-09-30", apply_smote=True), hierarchy
    )
    pipeline.run()


if __name__ == "__main__":
    main()


__all__ = ["PipelineConfig", "ClassificationPipeline", "main"]
