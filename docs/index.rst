===============================================
UniTSFM -- Unified Time Series Foundation Model Toolkit
===============================================

A unified, modular, PyTorch-based toolkit that wraps every major Time Series
Foundation Model (TimesFM, Chronos, MOIRAI, Lag-Llama, TinyTimeMixers) behind
a single ``model.fit(X) -> model.predict(h)`` API.

.. code-block:: python

    from uniftsm import UniTSFM

    model = UniTSFM(auto_select=True, ensemble=True)
    model.fit(electricity["load"])
    forecast = model.predict(horizon=168, return_intervals=True)


Features
========

* **5+ TSFMs, 1 API** - Chronos, TimesFM, MOIRAI, Lag-Llama, TTM -- same interface
* **Adaptive Patch Optimization** - differentiable gating over multiple patch scales
* **Uncertainty-Weighted Ensembling** - cross-architecture model fusion
* **Automatic Model Selection** - meta-feature based XGBoost classifier
* **Rigorous Evaluation** - MASE, sMAPE, CRPS, DM test, LaTeX-ready reports
* **CLI-First** - ``uniftsm forecast data.csv --horizon 90``


Novel Contributions
===================

1. `Adaptive Patch Size Optimization <api/pipeline.html#module-uniftsm.pipeline.patcher>`_
   A differentiable gating mechanism that learns the optimal patch size for each
   input series, outperforming fixed-size patching.

2. `Cross-Model Uncertainty Ensembling <api/ensemble.html#module-uniftsm.ensemble.uncertainty_weighted>`_
   Weights model contributions by their calibrated confidence, normalising
   heterogeneous uncertainty estimates across architectures.

3. `Meta-Learning for TSFM Selection <api/selection.html#module-uniftsm.selection.predictor>`_
   A lightweight XGBoost classifier over 15 meta-features that predicts the
   best TSFM without running any model.


.. toctree::
    :maxdepth: 2
    :caption: Getting Started

    tutorials/quickstart
    tutorials/installation


.. toctree::
    :maxdepth: 2
    :caption: API Reference

    api/core
    api/models
    api/pipeline
    api/ensemble
    api/selection
    api/evaluation
    api/datasets
    api/visualization
    api/cli


.. toctree::
    :maxdepth: 1
    :caption: Benchmarks

    benchmarks/overview


.. toctree::
    :maxdepth: 1
    :caption: Development

    contributing
    changelog


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
