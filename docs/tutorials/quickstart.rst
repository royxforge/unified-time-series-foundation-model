Quickstart
==========

This tutorial shows you how to get started with UniTSFM in under 5 minutes.


Installation
------------

Install the core library:

.. code-block:: bash

    pip install uniftsm

Then install one or more TSFM backends:

.. code-block:: bash

    pip install uniftsm[chronos]     # Amazon Chronos
    pip install uniftsm[timesfm]     # Google TimesFM
    pip install uniftsm[chronos,timesfm,moirai,lag_llama,ttm]  # all models


Basic Usage (Python API)
------------------------

.. code-block:: python

    import pandas as pd
    from uniftsm import UniTSFM

    # 1. Load your data
    y = pd.read_csv("electricity.csv", index_col=0, parse_dates=True).squeeze()

    # 2. Create a forecaster
    model = UniTSFM(model_name="chronos", model_size="base")

    # 3. Fit (prepares the series for forecasting)
    model.fit(y)

    # 4. Predict 24 steps ahead
    forecast = model.predict(horizon=24)
    print(forecast)
    #     mean       std
    # 0  102.3  2.1
    # 1  103.1  2.3
    # ...


Auto-Select + Ensemble
----------------------

.. code-block:: python

    model = UniTSFM(auto_select=True, ensemble=True)
    model.fit(y)
    forecast = model.predict(horizon=168, return_quantiles=True)

    # forecast is a dict of DataFrames, one per quantile
    print(forecast["q_0.5"].head())


CLI Usage
---------

.. code-block:: bash

    # Single model forecast
    uniftsm forecast data.csv --horizon 90 --model chronos

    # Auto-select + ensemble
    uniftsm forecast data.csv --horizon 168 --auto --ensemble

    # Benchmark multiple models
    uniftsm benchmark --datasets m4_hourly,electricity --models chronos,timesfm

    # Compare two models
    uniftsm compare chronos timesfm --dataset electricity

    # Start a prediction server
    uniftsm serve --model chronos --port 8080


Prediction Intervals
--------------------

.. code-block:: python

    model = UniTSFM(model_name="chronos")
    model.fit(y)

    lower, upper = model.predict_interval(horizon=24, alpha=0.05)
    # 95% prediction interval
