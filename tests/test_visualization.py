import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datalib_hk.visualization import DataVisualization


def test_bar_plot():
    data = {"Category": ["A", "B", "C"], "Values": [10, 20, 30]}
    df = pd.DataFrame(data)
    fig = DataVisualization.bar_plot(df, x_column="Category", y_column="Values", title="Bar Plot Test")
    assert isinstance(fig, plt.Figure)
    print("Bar Plot Test Passed!")


def test_histogram():
    data = np.random.randn(100)
    fig = DataVisualization.histogram(data, bins=15, title="Histogram Test")
    assert isinstance(fig, plt.Figure)
    print("Histogram Test Passed!")


def test_scatter_plot():
    data = {
        "X": np.random.rand(50),
        "Y": np.random.rand(50),
        "Group": np.random.choice(["Group1", "Group2"], size=50)
    }
    df = pd.DataFrame(data)
    fig = DataVisualization.scatter_plot(df, x_column="X", y_column="Y", hue="Group", title="Scatter Plot Test")
    assert isinstance(fig, plt.Figure)
    print("Scatter Plot Test Passed!")


def test_correlation_heatmap():
    data = np.random.rand(100, 4)
    df = pd.DataFrame(data, columns=["A", "B", "C", "D"])
    correlation_matrix = df.corr()
    fig = DataVisualization.correlation_heatmap(correlation_matrix, title="Correlation Heatmap Test")
    assert isinstance(fig, plt.Figure)
    print("Correlation Heatmap Test Passed!")


if __name__ == "__main__":
    test_bar_plot()
    test_histogram()
    test_scatter_plot()
    test_correlation_heatmap()
