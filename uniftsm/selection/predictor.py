"""Meta-classifier for automatic model selection (Novel Contribution #3).

A lightweight XGBoost classifier that predicts the best TSFM for a
given time series based on meta-features — without running any model.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from uniftsm.selection.meta_features import MetaFeatureExtractor

logger = logging.getLogger(__name__)


class ModelSelector:
    """Meta-classifier that predicts the best TSFM for a given series.

    Uses precomputed benchmark performance profiles and a lightweight
    XGBoost classifier trained on meta-features.

    Parameters
    ----------
    model_list:
        List of model names to consider.  If ``None``, all registered
        models are used.
    metric:
        Optimisation metric (e.g. ``"MASE"``, ``"sMAPE"``, ``"CRPS"``).
    """

    def __init__(
        self,
        model_list: list[str] | None = None,
        metric: str = "MASE",
    ) -> None:
        from uniftsm.core.registry import registry

        self.model_list = model_list or registry.list_models()
        self.metric = metric
        self._classifier: Any = None
        self._is_trained = False
        self._feature_names: list[str] | None = None

    def train(
        self,
        meta_features_df: pd.DataFrame,
        best_model_series: pd.Series,
    ) -> ModelSelector:
        """Train the meta-classifier.

        Args:
            meta_features_df: DataFrame where each row is a series and
                columns are meta-features (from :class:`MetaFeatureExtractor`).
            best_model_series: Series mapping each row to the name of the
                best-performing model.

        Returns:
            ``self`` for method chaining.
        """
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder

        self._feature_names = list(meta_features_df.columns)

        # Encode model names as labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(best_model_series)

        # Train XGBoost classifier
        self._classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        self._classifier.fit(meta_features_df.values, y_encoded)
        self._is_trained = True
        logger.info(
            "ModelSelector trained on %d series with %d meta-features.",
            len(meta_features_df),
            len(self._feature_names),
        )
        return self

    def predict(
        self,
        y: pd.Series,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Predict the best model(s) for a given series.

        Args:
            y: Univariate time series.
            top_k: Number of top recommendations to return.

        Returns:
            List of ``(model_name, confidence)`` tuples sorted descending
            by confidence.
        """
        if not self._is_trained or self._classifier is None:
            return self._rule_based_predict(y, top_k)

        features = MetaFeatureExtractor.extract(y)
        feature_vector = np.array(
            [features.get(f, 0.0) for f in (self._feature_names or [])]
        ).reshape(1, -1)

        probs = self._classifier.predict_proba(feature_vector)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]

        results = []
        for idx in top_indices:
            model_name = self._label_encoder.inverse_transform([idx])[0]
            results.append((model_name, float(probs[idx])))
        return results

    def _rule_based_predict(
        self,
        y: pd.Series,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Fallback rule-based prediction when classifier is not trained.

        Uses heuristics based on series length, frequency, and volatility.
        """
        from uniftsm.core.registry import registry

        features = MetaFeatureExtractor.extract(y)
        series_length = int(features["length"])
        freq_code = features["frequency"]

        return registry.suggest(
            series_length=series_length,
            frequency=str(freq_code),
            is_multivariate=False,
            needs_uncertainty=True,
            top_k=top_k,
        )
