import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, make_classification, make_blobs
from datalib_hk.advanced_analysis import AdvancedAnalysis

@pytest.fixture
def regression_data():
    """Generate synthetic data for regression testing."""
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
    y = pd.Series(y, name="target")
    return X, y

@pytest.fixture
def classification_data():
    """Generate synthetic data for classification testing."""
    X, y = make_classification(n_samples=100, n_features=3, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
    y = pd.Series(y, name="target")
    return X, y

@pytest.fixture
def clustering_data():
    """Generate synthetic data for clustering testing."""
    X, _ = make_blobs(n_samples=100, centers=3, n_features=3, random_state=42)
    X = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
    return X

def test_linear_regression(regression_data):
    X, y = regression_data
    results = AdvancedAnalysis.linear_regression(X, y)
    
    assert "model" in results
    assert "coefficients" in results
    assert "intercept" in results
    assert "mean_squared_error" in results
    assert "r_squared" in results
    assert results["mean_squared_error"] > 0
    assert 0 <= results["r_squared"] <= 1

def test_polynomial_regression(regression_data):
    X, y = regression_data
    results = AdvancedAnalysis.polynomial_regression(X, y, degree=2)
    
    assert "model" in results
    assert "mean_squared_error" in results
    assert "r_squared" in results
    assert results["mean_squared_error"] > 0
    assert 0 <= results["r_squared"] <= 1

def test_knn_classification(classification_data):
    X, y = classification_data
    results = AdvancedAnalysis.knn_classification(X, y, n_neighbors=3)
    
    assert "model" in results
    assert "accuracy" in results
    assert "predictions" in results
    assert 0 <= results["accuracy"] <= 1

def test_decision_tree_classification(classification_data):
    X, y = classification_data
    results = AdvancedAnalysis.decision_tree_classification(X, y, max_depth=3)
    
    assert "model" in results
    assert "accuracy" in results
    assert "feature_importance" in results
    assert 0 <= results["accuracy"] <= 1

def test_kmeans_clustering(clustering_data):
    X = clustering_data
    results = AdvancedAnalysis.kmeans_clustering(X, n_clusters=3)
    
    assert "model" in results
    assert "cluster_labels" in results
    assert "centroids" in results
    assert len(results["cluster_labels"]) == len(X)

def test_principal_component_analysis(clustering_data):
    X = clustering_data
    results = AdvancedAnalysis.principal_component_analysis(X, n_components=2)
    
    assert "transformed_data" in results
    assert "explained_variance_ratio" in results
    assert "cumulative_variance_ratio" in results
    assert "components" in results
    assert results["transformed_data"].shape[1] == 2
