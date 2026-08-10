Changelog
=========

Notable changes to UniTSFM are documented here.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_.

`0.1.0`_ -- 2026-07-24
-----------------------

Added
^^^^^

* Unified API wrapping 5 Time Series Foundation Models
* ``BaseForecaster`` abstract base class with ``fit()`` / ``predict()``
* ``ModelRegistry`` plugin architecture with ``@registry.register()``
* Adaptive Patch Size Optimization (Contribution 1)
* Cross-Model Uncertainty Ensembling (Contribution 2)
* Meta-Learning for TSFM Selection (Contribution 3)
* Evaluation: MASE, sMAPE, CRPS, NMAE, OWA, MRAE
* Statistical baselines: Naive, SeasonalNaive, ETS, ARIMA, Theta
* DM test and Wilcoxon signed-rank significance tests
* LaTeX-ready comparison report generation
* Dataset loaders: Monash, LOTSA, M3/M4, synthetic
* Forecast, uncertainty, comparison, and attention visualizations
* CLI with 4 subcommands: ``forecast``, ``benchmark``, ``compare``, ``serve``
* 330+ unit, integration, and property-based tests
* Pipeline: adaptive patching, scaling, tokenization, frequency detection,
  missing value imputation
* Benchmark framework with JSON/CSV serialization
* Conformal prediction intervals
* Fully typed API with PEP 484 annotations

.. _0.1.0: https://github.com/royxforge/uniftsm/releases/tag/v0.1.0
