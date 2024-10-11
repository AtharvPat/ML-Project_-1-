import pytest
import numpy as np
import pandas as pd
import sys 

sys.path.insert(1, 'E:/IIT CHICAGO/SEMESTER_1/MACHINE LEARNING/PROJECT')

from Models.ElasticNet import ElasticNetModel
from Models.ElasticNet import ElasticNetModelResults
from Models.Data_Gen import DataGenerator
from Models.Data_Gen import ProfessorData
from Test.Test_Model import *


@pytest.fixture
def generate_data():
    """Fixture to generate data."""
    data_gen = DataGenerator(rows= 1000, cols= 10, noise= 0.3, seed= 10)
    X, y = data_gen.gen_data()
    return X, y


def test_data_generation(generate_data):
    """Test if data generation returns the correct shapes."""
    X, y =generate_data
    assert X.shape == (1000, 10), f"Expected X shape (1000, 10), but got {X.shape} "
    assert y.shape == (1000,), f"Expected y shape (1000,), but got {y.shape}"


def test_train_test_split( generate_data):
    """Test train-test split testing."""
    X, y = generate_data
    X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.2)

    assert X_train.shape[0] == 800, f"Expected 80 training samples, but got {X_train.shape[0]}"
    assert X_test.shape[0] == 200, f"Expected 20 test samples, but got {X_test.shape[0]}"
    assert y_train.shape[0] == 800, f"Expected 80 training labels, but got {y_train.shape[0]}"
    assert y_test.shape[0] == 200, f"Expected 20 test labels, but got {y_test.shape[0]}"


def test_standardize(generate_data):
    """Test standardize"""
    X, _ = generate_data
    X_train, X_test, _, _ = train_test_split(X, np.arange(1000), test_size=0.2)

    X_train_std, X_test_std = standardize(X_train, X_test)
    
    assert np.allclose(X_train_std.mean(axis=0), 0, atol=1e-7), "Expected mean close to 0 after standardization for each column"
    assert np.allclose(X_train_std.std(axis=0), 1, atol=1e-7), "Expected std close to 1 after standardization for each column"



@pytest.fixture
def elasticnet_model():

    return ElasticNetModel(alpha=0.01,penalty_ratio=0.1,learning_rate=0.001,  iterations=1000)


def test_elasticnet_model_fit_predict(generate_data, elasticnet_model):
    """Test ElasticNetModel fitting and prediction."""
    X, y = generate_data
    X_train,X_test,  y_train,y_test = train_test_split(X, y,test_size=0.2)
    X_train_std,X_test_std = standardize(X_train,  X_test)
    
    elasticnet_model.fit(X_train_std.values,  y_train.values)
    
    y_pred = elasticnet_model.predict(X_test_std.values)
    
    assert y_pred.shape == y_test.shape, "   Predictions matchs the shape of y_test"


def test_metrics(generate_data,  elasticnet_model):
    """Test MSE, MAE, and R2 metrics."""
    X, y = generate_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_std, X_test_std = standardize(X_train, X_test)
    
    elasticnet_model.fit( X_train_std.values,y_train.values)
    y_pred = elasticnet_model.predict(  X_test_std.values)
    
    comparison_df = pd.DataFrame({
        'Actual Values': y_test.values,
        'Predicted Values': y_pred
    })
    comparison_df["Difference"] = comparison_df['Actual Values'] - comparison_df['Predicted Values']
    
    mse = np.square(comparison_df["Difference"]).mean()
    assert mse >= 0, "MSE should be a non-negative value"

    mae = np.abs(comparison_df['Difference']).mean()
    assert mae >= 0, "MAE should be a non-negative value"

    results = ElasticNetModelResults(y_test=y_test.values, y_pred=y_pred)
    r2 = results.r2_score(y_test.values, y_pred)
    assert -1 <= r2 <= 1, "R² should be between -1 and 1"


