# Benchmarks

This directory contains the UniTSFM benchmarking infrastructure and its
artifacts.  All scripts are reproducible: they fix random seeds and write
results as CSV **and** LaTeX.

## What each artifact means (honesty first)

| File | Content | How it was produced |
|---|---|---|
| `benchmark_results.csv` | Per model × dataset metrics (MASE, sMAPE, CRPS, RMSE, MAE, time) | `run_all.py --quick` — **real** 7-dataset Monash run: all 5 TSFMs (chronos, timesfm, moirai, ttm, lag_llama) × 2 series per dataset on CPU |
| `benchmark_table.tex` | LaTeX-ready comparison table (one row per metric) | Generated from the same results via `uniftsm.evaluation.report.latex_table` |
| `summary.csv` | Mean metrics per model × dataset | Same run |
| `benchmark_mase.png` | Faceted MASE bar chart (one panel per dataset, best model hatched) | `plot_results.py` from the same run |
| `benchmark_crps.png` | Faceted CRPS bar chart (probabilistic accuracy) | `plot_results.py --metric CRPS` from the same run |
| `synthetic_benchmark.csv/.json`, `synthetic_summary.csv` | Statistical baselines on synthetic series | `synthetic_benchmark.py` |
| `mock_benchmark_results.csv/.json`, `mock_summary.csv` | CI smoke test with `MockForecaster` | `run_mock_benchmark.py` — validates the pipeline, **not** real model performance |

## Reproducing the full paper benchmark

The README's headline results (71 % ensemble win rate, 18 % lower CRPS)
correspond to the full 7-dataset Monash run over all 5 TSFMs:

```bash
python benchmarks/run_all.py --quick          # 2 series per dataset (smoke)
python benchmarks/run_all.py                  # 10 series per dataset (full)
python benchmarks/run_all.py --dataset electricity
python benchmarks/plot_results.py             # regenerate benchmark_mase.png
python benchmarks/plot_results.py --metric CRPS --output results/benchmark_crps.png
```

Requirements: all TSFM backends installed (`pip install "uniftsm[all]"`),
HuggingFace authentication for gated checkpoints (`huggingface-cli login`),
and a GPU is strongly recommended — the full run is otherwise extremely
slow.  `run_all.py` writes `benchmarks/results/benchmark_results.csv`
and `benchmarks/results/benchmark_table.tex` (overwriting the
preliminary artifacts above).

### Latest verified results (2026-08, `run_all.py --quick`, CPU, all 5 TSFMs)

Best model per dataset by MASE — **TimesFM wins 6/7, TTM wins m4_hourly**:

| Dataset | Best model | MASE |
|---|---|---|
| electricity | timesfm | 0.394 |
| exchange_rate | timesfm | 2.765 |
| influenza (NN5 weekly) | timesfm | 0.945 |
| m4_hourly | **ttm** | 0.462 |
| solar | timesfm | 0.050 |
| traffic | timesfm | 0.223 |
| weather | timesfm | 0.613 |

Mean MASE across datasets: timesfm **0.955** < ttm 1.648 < chronos 2.468
< moirai 3.927 < lag_llama 14.115.  Lag-Llama is now runtime-verified too
(installed from the git repo; its weak exchange-rate MASE is consistent
with the zero-shot LLM forecaster's known behaviour on high-entropy FX
series).

## Lightweight runs (no model downloads)

```bash
python benchmarks/synthetic_benchmark.py --quick   # baselines only, ~seconds
python benchmarks/run_mini_benchmark.py            # baselines + Chronos-tiny, ~minutes
python benchmarks/run_mock_benchmark.py            # mock models, CI-safe
```

## Tests

`test_benchmark.py` validates the metric computation, LaTeX generation,
and reproducibility guarantees used by the runners.
