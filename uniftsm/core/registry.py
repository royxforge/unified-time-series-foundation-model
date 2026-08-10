"""Central registry of all available TSFM models.

New models are added via the ``@registry.register()`` decorator — no core
code changes needed.  This is a true plugin architecture.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uniftsm.core.base import BaseForecaster

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry of all available TSFM model classes.

    Usage::

        from uniftsm.core.registry import registry

        @registry.register("my_model", metadata={"paper": "..."})
        class MyModel(BaseForecaster):
            ...

        cls = registry.get("my_model")
        model = cls()
    """

    def __init__(self) -> None:
        self._models: dict[str, type[BaseForecaster]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    # ── Registration ─────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Callable[[type[BaseForecaster]], type[BaseForecaster]]:
        """Decorator that registers a :class:`BaseForecaster` subclass.

        Args:
            name: Canonical model name (e.g. ``"chronos"``, ``"timesfm"``).
            metadata: Optional dictionary of model properties (family,
                supported sizes, context limits, paper reference, etc.).

        Returns:
            The original class unchanged.
        """

        def wrapper(
            model_class: type[BaseForecaster],
        ) -> type[BaseForecaster]:
            if name in self._models:
                logger.warning(
                    "Overwriting registered model '%s' (was %s).",
                    name,
                    self._models[name].__name__,
                )
            self._models[name] = model_class
            self._metadata[name] = metadata or {}
            logger.debug("Registered model '%s': %s", name, model_class.__name__)
            return model_class

        return wrapper

    # ── Lookup ───────────────────────────────────────────────────────────

    def get(self, name: str) -> type[BaseForecaster]:
        """Retrieve a registered model class by name.

        Raises
        ------
        ValueError
            If the name is not registered.
        """
        if name not in self._models:
            available = ", ".join(sorted(self._models))
            raise ValueError(f"Unknown model: '{name}'. " f"Available models: [{available}]")
        return self._models[name]

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Return metadata for a registered model."""
        self.get(name)  # raises if unknown
        return self._metadata.get(name, {})

    def list_models(self) -> list[str]:
        """Return all registered model names (sorted alphabetically)."""
        return sorted(self._models)

    def list_with_metadata(self) -> list[dict[str, Any]]:
        """Return all models with their metadata for display / CLI."""
        return [{"name": name, **self._metadata.get(name, {})} for name in sorted(self._models)]

    # ── Suggestion / auto-selection ──────────────────────────────────────

    def suggest(
        self,
        series_length: int,
        frequency: str,
        is_multivariate: bool,
        needs_uncertainty: bool,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Suggest the best models for a given series, with confidence scores.

        This is a heuristic rule-based system.  The full meta-learning
        predictor is in :mod:`uniftsm.selection`.

        Args:
            series_length: Number of observed time steps.
            frequency: Pandas frequency alias.
            is_multivariate: Whether the series has multiple channels.
            needs_uncertainty: Whether probabilistic output is required.
            top_k: Number of suggestions to return.

        Returns:
            List of ``(model_name, score)`` tuples sorted descending by score.
        """
        scores: dict[str, float] = {}

        for name in self._models:
            meta = self._metadata.get(name, {})
            score = 0.0

            # Length compatibility
            min_len = meta.get("min_length", 0)
            max_ctx = meta.get("max_context_length", float("inf"))
            if series_length < min_len:
                continue  # incompatible
            if series_length <= max_ctx:
                score += 2.0
            else:
                score += 0.5  # will truncate

            # Multivariate support
            supports_multi = meta.get("supports_multivariate", False)
            if is_multivariate and supports_multi:
                score += 3.0
            elif is_multivariate and not supports_multi:
                score -= 1.0

            # Probabilistic support
            supports_prob = meta.get("supports_probabilistic", False)
            if needs_uncertainty and supports_prob:
                score += 2.0
            elif needs_uncertainty and not supports_prob:
                score -= 0.5

            # Preference for smaller models on small data
            if series_length < 256 and meta.get("family") in ("chronos", "ttm"):
                score += 1.0

            scores[name] = score

        sorted_models = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_models[:top_k]

    # ── Inspection ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def __repr__(self) -> str:
        models = ", ".join(self.list_models())
        return f"ModelRegistry({len(self)} models: [{models}])"


# Module-level singleton.
registry: ModelRegistry = ModelRegistry()
