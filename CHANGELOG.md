# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Unified API wrapping 5 Time Series Foundation Models (Chronos, TimesFM, MOIRAI, Lag-Llama, TinyTimeMixers)
- `BaseForecaster` abstract base class with `fit()` / `predict()` interface
- `ModelRegistry` plugin architecture with `@registry.register()` decorator
- Adaptive Patch Size Optimization module (novel contribution 1)
- Cross-Model Uncertainty Ensembling module (novel contribution 2)
- Meta-Learning for TSFM Selection module (novel contribution 3)
- Evaluation framework: MASE, sMAPE, CRPS, NMAE, OWA, MRAE metrics
- Statistical baselines: Naive, SeasonalNaive, ETS, ARIMA, Theta
- Diebold-Mariano test and Wilcoxon signed-rank significance tests
- LaTeX-ready comparison report generation
- Dataset loaders: Monash Forecasting Archive, LOTSA, M3/M4, synthetic
- Visualization: forecast plots, fan charts, model comparison, attention maps
- CLI with 4 subcommands: `forecast`, `benchmark`, `compare`, `serve`
- Comprehensive test suite: 330+ unit, integration, and property-based tests
- Editable pipeline: patching, scaling, tokenization, frequency detection, imputation
- Benchmark framework with synthetic, mock, and real-dataset runners
- Reproducible benchmarking with fixed seeds and result serialization
- Conformal prediction intervals for ensemble forecasts
- `pyproject.toml` with optional dependency groups for each TSFM
- `python -m uniftsm` entry point (`uniftsm/__main__.py`)
- Real preliminary benchmark runner: `benchmarks/run_mini_benchmark.py`
- `benchmarks/README.md` documenting artifacts and reproduction steps
- CI workflow (`.github/workflows/ci.yml`), pre-commit hooks, and `.gitignore`
- `ruff`, `black`, `mypy`, and `pytest` configuration blocks in `pyproject.toml`

### Changed
- `uniftsm/__init__.py` now imports `uniftsm.models` so all TSFM wrappers are
  registered on package import (previously the registry was empty until the
  user imported a model module manually)
- `hypothesis` added to the `dev` extra so the property-based test suite is
  runnable via `pip install -e ".[dev]"`
- `BaseForecaster.__init__` gives `model_name` an empty-string default so
  registry-based construction type-checks cleanly
- Benchmark section of the README now clearly separates reported paper
  results from the preliminary reproducible results shipped in this repo

### Deprecated

### Removed

### Fixed
- Critical: `UniTSFM(model_name=...)`, `auto_select=True`, and `ensemble=True`
  crashed with `ValueError: Unknown model` on a fresh import
- Wrong HuggingFace repository IDs: TimesFM now resolves to
  `google/timesfm-2.5-200m-pytorch`, TTM to
  `ibm-granite/granite-timeseries-ttm-r2`, and Lag-Llama to
  `time-series-foundation-models/lag-llama` (previous IDs were gated or
  non-existent, so those models could never load)
- TTM wrapper now loads via `TinyTimeMixerForPrediction` with 3-D
  `past_values` and truncates the model's fixed prediction length to the
  requested horizon (verified end-to-end on CPU)
- TimesFM wrapper now uses the 2.x API: `from_pretrained` → `compile(
  ForecastConfig(...))` → `forecast(horizon, [series])`, with std derived
  from the model's quantile output
- Lag-Llama wrapper rewritten around the documented `LagLlamaEstimator`
  API (git-installed package; the previous raw `LlamaForCausalLM.generate`
  call could not work)
- MOIRAI wrapper rewritten for the uni2ts 1.x API
  (`MoiraiModule.from_pretrained` + `MoiraiForecast`/GluonTS predictor; the
  previous `from uni2ts import MoiraiModel` path never worked on 1.x) and
  per-channel inverse-scaling fixed (verified end-to-end on CPU)
- Uncertainty-weighted ensemble now skips members whose backend cannot run
  (e.g. partial installs) with a warning instead of failing entirely
- `_hf_model_dir()` used a removed `transformers` API; now resolves the
  HuggingFace cache via `huggingface_hub`
- 61 lint errors and 31 type errors across the package resolved (ruff + mypy
  now pass with zero findings)
- `float64` roundtrip precision in the scaler property tests (tolerance was
  tighter than machine epsilon for data of magnitude 1e4)
- `RuntimeWarning`s from sample-entropy log(0) and kurtosis catastrophic
  cancellation on (near-)constant series
- Scaler / tokenizer return typing made explicit for numpy stubs
- Monash loader: the `influenza` entry pointed at the FRED-MD Zenodo record
  with a pattern that could never match; it now resolves to the NN5 Weekly
  dataset (record 4656125, `nn5_weekly_dataset.zip`) and uses the
  non-deprecated `W` frequency
- Frequency detection normalises aliases that pandas infers but downstream
  rejects (`"10min"` → `"10T"`) and canonicalises the lowercase aliases
  deprecated by pandas 2.2+ (`"w"` → `"W"`); this unblocked the Monash
  `solar` (10-minutely) and `influenza` (weekly) benchmarks, which
  previously crashed every model with `invalid unit abbreviation`
- `BaseForecaster._make_index` now steps the forecast index via
  `pd.tseries.frequencies.to_offset`, so multiplier aliases (`"10T"`,
  `"2H"`) and anchored offsets (`"W-SUN"`) no longer raise
  `ValueError: invalid unit abbreviation` in `pd.Timedelta(1, unit=...)`
- TTM wrapper left-pads contexts shorter than the model's fixed 512-step
  requirement instead of failing with
  `Context length in past_values is shorter than TTM context_length`
  (e.g. the short NN5 influenza series)
- `SeriesScaler` robust method floors a degenerate / subnormal IQR so
  pathological inputs (single outlier among identical values) stay finite
  and invertible instead of overflowing to `inf`
- `latex_table` emits one row per metric with a `Dataset & Metric` header
  (previously every metric × model value was crammed into a single row
  producing structurally invalid LaTeX)
- Real 7-dataset Monash benchmark results (2 series × 5 verified TSFMs per
  dataset) are shipped in `benchmarks/results/`
- Lag-Llama wrapper rewritten to match the installed git-pinned package:
  import path `lag_llama.gluon.estimator` (not `estimators`), the
  `create_lightning_module`/`create_transformation`/`create_predictor`
  assembly path (the high-level `get_predictor`/`predict` helpers do not
  exist in that fork), `distr_output` passed as the string `"studentT"`,
  `time_feat`/`scaling`/architecture read from the checkpoint's
  `hyper_parameters`, `lags_seq` taken from the checkpoint (the
  GluonTS frequency→lag derivation raises "invalid frequency" on
  pandas 2.x), `torch.load(weights_only=False)` workaround for the
  PyTorch 2.6 default change, and the dataset timestamps truncated to
  match the scaled context.  Verified end-to-end on CPU (6 s forecast).

### Security

## [0.1.0] - 2026-07-24

### Added
- Initial project scaffold and directory structure
- Core module: `base.py`, `registry.py`, `config.py`, `exceptions.py`
- Chronos model wrapper (amazon-science/chronos-forecasting)
- TimesFM model wrapper (google/timesfm)
- MOIRAI model wrapper (salesforce/moirai)
- Lag-Llama model wrapper (time-series-foundation-models/lag-llama)
- TTM model wrapper (IBM/TinyTimeMixers)
- Shared model download and caching logic (`_base_impl.py`)
- All pipeline, ensemble, selection, evaluation, dataset, and visualization modules
- CLI with Click-based argument parsing
- Test framework with pytest, hypothesis, and property-based tests
- Project documentation: README, CONTRIBUTING, LICENSE
- Benchmark infrastructure with JSON/CSV result serialization
- Paper: LaTeX structure with novel contributions described
