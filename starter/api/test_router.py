"""
Unit tests for the API router endpoints.
"""
from fastapi.testclient import TestClient
from fastapi import FastAPI
from api.router import router

# Create a test app
app = FastAPI()
app.include_router(router)

# Create test client
client = TestClient(app)


def test_get_root():
    """
    Test the GET request on the root endpoint.
    Tests both status code and response content.
    """
    response = client.get("/")
    
    # Test status code
    assert response.status_code == 200
    
    # Test response content
    response_json = response.json()
    assert "message" in response_json
    assert response_json["message"] == "Welcome to the Census Income Prediction API!"
    assert isinstance(response_json["message"], str)


def test_post_predict_below_50k():
    """
    Test POST request that predicts income <=50K.
    Tests both status code and that prediction is <=50K.
    """
    # Sample data for someone likely earning <=50K
    # Young person, low education, entry-level occupation
    data = {
        "age": 19,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 25,
        "native-country": "United-States"
    }
    
    response = client.post("/predict", json=data)

    # Test status code
    assert response.status_code == 200
    
    # Test response content
    response_json = response.json()
    assert "prediction" in response_json
    assert isinstance(response_json["prediction"], str)
    # This profile should predict <=50K
    assert response_json["prediction"] == "<=50K"


def test_post_predict_above_50k():
    """
    Test POST request that predicts income >50K.
    Tests both status code and that prediction is >50K.
    """
    # Sample data for someone likely earning >50K
    # Older, highly educated, professional, married, high capital gains
    data = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }
    
    response = client.post("/predict", json=data)
    
    # Test status code
    assert response.status_code == 200
    
    # Test response content
    response_json = response.json()
    assert "prediction" in response_json
    assert isinstance(response_json["prediction"], str)
    # This profile should predict >50K
    assert response_json["prediction"] == ">50K"


def test_post_predict_malformed_data():
    """
    Test POST request with malformed data that causes an error.
    Tests that the API returns 422 for validation errors.
    """
    # Malformed data - missing required fields
    data = {
        "age": 30,
        "workclass": "Private",
        # Missing many required fields
    }
    
    response = client.post("/predict", json=data)
    
    # Test status code - FastAPI returns 422 for validation errors
    assert response.status_code == 422
    
    # Test response contains error detail
    response_json = response.json()
    assert "detail" in response_json
