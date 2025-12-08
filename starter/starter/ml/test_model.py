"""
Unit tests for the model module.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model import train_model, compute_model_metrics, inference


def test_train_model():
    """Test that train_model returns a fitted RandomForestClassifier."""
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    hyperparameters = {
        "n_estimators": 10,
        "max_depth": 5,
        "random_state": 42
    }
    
    model = train_model(X_train, y_train, hyperparameters)
    
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 10
    assert hasattr(model, 'n_features_in_')


def test_compute_model_metrics():
    """Test that compute_model_metrics calculates correct metrics."""
    y = np.array([0, 1, 0, 1, 1, 0])
    preds = np.array([0, 1, 0, 1, 1, 0])
    
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_inference():
    """Test that inference returns predictions with correct shape."""
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    hyperparameters = {
        "n_estimators": 10,
        "random_state": 42
    }
    model = train_model(X_train, y_train, hyperparameters)
    X_test = np.random.rand(20, 5)
    
    preds = inference(model, X_test)
    
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 20
    assert all(pred in [0, 1] for pred in preds)


def test_model_metrics_with_partial_accuracy():
    """Test metrics calculation with partial accuracy."""
    y = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])
    
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    assert 0.0 < precision <= 1.0
    assert 0.0 < recall <= 1.0
    assert 0.0 < fbeta <= 1.0
    assert abs(precision - 0.6667) < 0.001
    assert recall == 1.0
