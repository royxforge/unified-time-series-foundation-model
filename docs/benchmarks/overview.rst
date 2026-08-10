Benchmarks Overview
===================

UniTSFM includes a comprehensive benchmarking framework for evaluating
TSFMs across diverse datasets and metrics.


Benchmark Infrastructure
------------------------

Benchmark scripts are located in ``benchmarks/``:

.. code-block:: text

    benchmarks/
    ├── run_all.py               # Full benchmark suite
    ├── synthetic_benchmark.py   # Statistical baselines only (no TSFM deps)
    ├── run_mock_benchmark.py    # CI-safe mock forecaster benchmarks
    ├── test_benchmark.py        # Validation tests
    └── results/                 # Pre-computed results
        ├── synthetic_benchmark.csv
        ├── synthetic_benchmark.json
        ├── synthetic_summary.csv
        ├── mock_benchmark_results.csv
        ├── mock_benchmark_results.json
        └── mock_summary.csv


Running Benchmarks
------------------

.. code-block:: bash

    # Synthetic benchmark (statistical baselines only, no TSFM deps)
    python benchmarks/synthetic_benchmark.py

    # CI-safe benchmark (uses MockForecaster)
    python benchmarks/run_mock_benchmark.py

    # Full benchmark with real models (requires TSFM packages)
    # See benchmarks/run_all.py for configuration
    python benchmarks/run_all.py


Datasets
--------

Benchmarks draw from these sources:

* **Monash Forecasting Archive** -- 30+ datasets across domains (energy,
  traffic, weather, finance, health)
* **M3 / M4 Competition** -- 1003/100000 series from forecasting
  competitions
* **LOTSA** -- Large-scale open time series archive for pre-training
  evaluation
* **Synthetic** -- Controlled test cases with known ground truth

Metrics
-------

| Metric     | What it measures               | Preferred direction |
|------------|--------------------------------|---------------------|
| MASE       | Scale-adjusted accuracy        | Lower is better     |
| sMAPE      | Symmetric percentage error     | Lower is better     |
| CRPS       | Probabilistic calibration      | Lower is better     |
| Coverage   | Prediction interval reliability | Close to nominal    |
| OWA        | Combined MASE + sMAPE score    | Lower is better     |

Reproducibility
---------------

All benchmarks use a fixed random seed (``seed=42``) and save results to
``benchmarks/results/`` in both CSV and JSON formats.  The seed can be
overridden via the ``--seed`` CLI argument or configuration.
