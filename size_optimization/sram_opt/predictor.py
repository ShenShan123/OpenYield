"""Rehydrate the external SramOpt TabPFN context without bundling its data."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .contract import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, feature_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle(bundle: Mapping[str, Any]) -> None:
    """Reject incompatible features instead of silently rearranging them."""
    required = {
        "feature_columns", "categorical_columns", "categorical_feature_indices",
        "ordinal_encoder", "numeric_medians", "x_train", "y_train_model",
        "checkpoint", "n_estimators", "target_transform", "feasibility_target",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"SramOpt TabPFN bundle is missing fields: {missing}")
    if tuple(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("SramOpt bundle must use the original ordered 26-feature contract")
    if tuple(bundle["categorical_columns"]) != CATEGORICAL_COLUMNS:
        raise ValueError("SramOpt bundle has incompatible categorical features")
    expected = [FEATURE_COLUMNS.index(name) for name in CATEGORICAL_COLUMNS]
    if list(bundle["categorical_feature_indices"]) != expected:
        raise ValueError("SramOpt bundle has incompatible categorical indices")
    x_train = np.asarray(bundle["x_train"])
    y_train = np.asarray(bundle["y_train_model"])
    if x_train.ndim != 2 or x_train.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("SramOpt bundle x_train must have 26 feature columns")
    if not len(x_train) or y_train.shape != (len(x_train),):
        raise ValueError("SramOpt bundle contains an invalid training context")
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise ValueError("SramOpt bundle training context must be finite")
    if bundle["target_transform"] not in ("raw", "jeffreys_logit"):
        raise ValueError("unsupported SramOpt yield target transform")
    target = float(bundle["feasibility_target"])
    if not np.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("bundle feasibility target must be in [0, 1]")


def encode(frame: Any, bundle: Mapping[str, Any]) -> np.ndarray:
    """Use the original SramOpt categorical encoder and numeric medians."""
    import pandas as pd

    categorical = bundle["ordinal_encoder"].transform(
        frame.loc[:, list(CATEGORICAL_COLUMNS)].astype(str)
    )
    categorical_index = {name: index for index, name in enumerate(CATEGORICAL_COLUMNS)}
    result = np.empty((len(frame), len(FEATURE_COLUMNS)), dtype=np.float32)
    for index, name in enumerate(FEATURE_COLUMNS):
        if name in categorical_index:
            result[:, index] = categorical[:, categorical_index[name]]
        else:
            values = pd.to_numeric(frame[name], errors="coerce").fillna(
                float(bundle["numeric_medians"][name])
            )
            result[:, index] = values.to_numpy(dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("SramOpt encoded features must be finite")
    return result


class TabPFNYieldPredictor:
    """Load a user-supplied trusted joblib bundle and a local checkpoint."""

    def __init__(
        self,
        bundle_path: Optional[str],
        *,
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        seed: int = 20260717,
    ) -> None:
        if not bundle_path:
            raise FileNotFoundError(
                "SramOpt needs an external trained TabPFN bundle: pass --bundle PATH. "
                "Training data and model files were intentionally not merged. "
                "Use --check for a data-free integration check."
            )
        path = Path(bundle_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SramOpt TabPFN bundle not found: {path}")

        import joblib

        self.bundle = joblib.load(path)
        validate_bundle(self.bundle)
        model_path = Path(checkpoint or str(self.bundle["checkpoint"])).expanduser()
        if not model_path.is_absolute():
            model_path = path.parent / model_path
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"TabPFN checkpoint not found: {model_path}; pass --checkpoint PATH"
            )
        expected_hash = self.bundle.get("checkpoint_sha256")
        if expected_hash and _sha256(model_path) != str(expected_hash):
            raise ValueError("TabPFN checkpoint hash differs from the training bundle")

        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        from tabpfn import TabPFNRegressor

        self.model = TabPFNRegressor(
            n_estimators=int(self.bundle["n_estimators"]),
            categorical_features_indices=self.bundle["categorical_feature_indices"],
            model_path=model_path,
            device=device,
            ignore_pretraining_limits=True,
            fit_mode="fit_preprocessors",
            random_state=int(seed),
            n_preprocessing_jobs=1,
        )
        self.model.fit(self.bundle["x_train"], self.bundle["y_train_model"])
        self.feasibility_target = float(self.bundle["feasibility_target"])
        self.provenance: Dict[str, Any] = {
            "bundle_path": str(path),
            "bundle_sha256": _sha256(path),
            "checkpoint_path": str(model_path),
            "target_column": self.bundle.get("target_column"),
            "variation_contract": self.bundle.get("variation_contract"),
            "compiler_calibration": "external bundle; V2.2.6 calibration not established",
        }

    def predict(self, designs: Sequence[Mapping[str, Any]]) -> np.ndarray:
        frame = feature_rows(designs)
        latent = np.asarray(self.model.predict(encode(frame, self.bundle)), dtype=float)
        if latent.shape != (len(frame),) or not np.isfinite(latent).all():
            raise ValueError("TabPFN returned invalid yield predictions")
        if self.bundle["target_transform"] == "raw":
            return np.clip(latent, 0.0, 1.0)
        return 1.0 / (1.0 + np.exp(-np.clip(latent, -30.0, 30.0)))
