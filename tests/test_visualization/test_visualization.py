"""Tests for visualization modules.

Uses matplotlib's Agg backend (no display required).
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd


class TestForecastPlot:
    def test_returns_figure(self, simple_series):
        import matplotlib.pyplot as plt

        from uniftsm.visualization.forecasts import plot_forecast

        y_pred = pd.DataFrame({"mean": np.zeros(10), "std": np.ones(10)})
        fig = plot_forecast(simple_series, y_pred)
        assert fig is not None
        plt.close(fig)

    def test_with_ground_truth(self, simple_series):
        import matplotlib.pyplot as plt

        from uniftsm.visualization.forecasts import plot_forecast

        y_pred = pd.DataFrame({"mean": np.zeros(10), "std": np.ones(10)})
        y_true = pd.Series(np.zeros(10))
        fig = plot_forecast(simple_series, y_pred, y_true=y_true)
        assert fig is not None
        plt.close(fig)


class TestUncertaintyFanChart:
    def test_fan_chart(self):
        import matplotlib.pyplot as plt

        from uniftsm.visualization.uncertainty import plot_uncertainty_fan

        idx = pd.RangeIndex(20)
        forecast = {
            "q_0.1": pd.DataFrame({"forecast": np.zeros(20)}, index=idx),
            "q_0.5": pd.DataFrame({"forecast": np.zeros(20)}, index=idx),
            "q_0.9": pd.DataFrame({"forecast": np.zeros(20)}, index=idx),
        }
        fig = plot_uncertainty_fan(forecast)
        assert fig is not None
        plt.close(fig)

    def test_with_ground_truth(self):
        import matplotlib.pyplot as plt

        from uniftsm.visualization.uncertainty import plot_uncertainty_fan

        idx = pd.RangeIndex(20)
        forecast = {
            "q_0.1": pd.DataFrame({"forecast": np.zeros(20)}, index=idx),
            "q_0.5": pd.DataFrame({"forecast": np.zeros(20)}, index=idx),
            "q_0.9": pd.DataFrame({"forecast": np.zeros(20)}, index=idx),
        }
        y_true = pd.Series(np.zeros(20), index=idx)
        fig = plot_uncertainty_fan(forecast, y_true=y_true)
        assert fig is not None
        plt.close(fig)


class TestComparisonPlot:
    def test_model_comparison(self, simple_series):
        import matplotlib.pyplot as plt

        from uniftsm.visualization.comparison import plot_model_comparison

        idx = pd.RangeIndex(10)
        forecasts = {
            "model_a": pd.DataFrame({"mean": np.zeros(10)}, index=idx),
            "model_b": pd.DataFrame({"mean": np.ones(10)}, index=idx),
        }
        fig = plot_model_comparison(simple_series, forecasts)
        assert fig is not None
        plt.close(fig)

    def test_rank_heatmap(self):
        import matplotlib.pyplot as plt

        from uniftsm.visualization.comparison import plot_rank_heatmap

        rank_matrix = pd.DataFrame(
            {
                "chronos": [1, 3, 2],
                "timesfm": [2, 1, 3],
                "moirai": [3, 2, 1],
            },
            index=["dataset_a", "dataset_b", "dataset_c"],
        )
        fig = plot_rank_heatmap(rank_matrix)
        assert fig is not None
        plt.close(fig)


class TestAttentionPlot:
    def test_attention_map_2d(self):
        import matplotlib.pyplot as plt

        from uniftsm.visualization.attention import plot_attention_map

        weights = np.random.rand(8, 8)
        fig = plot_attention_map(weights)
        assert fig is not None
        plt.close(fig)

    def test_attention_map_3d(self):
        import matplotlib.pyplot as plt

        from uniftsm.visualization.attention import plot_attention_map

        weights = np.random.rand(4, 8, 8)  # 4 heads
        fig = plot_attention_map(weights)
        assert fig is not None
        plt.close(fig)

    def test_attention_rollout(self):
        import matplotlib.pyplot as plt

        from uniftsm.visualization.attention import plot_attention_rollout

        matrices = [np.random.rand(8, 8) for _ in range(3)]
        fig = plot_attention_rollout(matrices)
        assert fig is not None
        plt.close(fig)
