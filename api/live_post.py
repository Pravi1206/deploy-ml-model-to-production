"""
Script to POST to the Census Income Prediction API and display results.
"""
import requests
import json


def post_to_api(url: str, data: dict) -> tuple:
    """
    Send a POST request to the API and return the result and status code.
    
    Args:
        url: API endpoint URL
        data: Dictionary containing census data
        
    Returns:
        tuple: (status_code, response_json)
    """
    try:
        response = requests.post(url, json=data)
        return response.status_code, response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None, None


def get_from_api(url: str) -> tuple:
    """
    Send a GET request to the API and return the result and status code.
    
    Args:
        url: API endpoint URL
        
    Returns:
        tuple: (status_code, response_json)
    """
    try:
        response = requests.get(url)
        return response.status_code, response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None, None


def main():
    """
    Main function to test the API with sample data.
    """
    # API endpoints
    api_root = "https://deploy-ml-model-to-production-1.onrender.com/"  # Replace with actual URL if needed
    api_predict = "https://deploy-ml-model-to-production-1.onrender.com/predict"  # Replace with actual URL if needed
    
    print("=" * 80)
    print("Census Income Prediction API - Live Request Test")
    print("=" * 80)
    
    # Test 0: GET request to root
    print("\n[Test 0] GET Request to Root Endpoint")
    print("-" * 80)
    print(f"URL: {api_root}")
    status_code, result = get_from_api(api_root)
    
    if status_code:
        print(f"\nStatus Code: {status_code}")
        print(f"Response: {json.dumps(result, indent=2)}")
    else:
        print("\nFailed to get response from API")
    
    # Sample data 1: Profile likely earning <=50K
    data_low_income = {
        "age": 22,
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
        "hours-per-week": 30,
        "native-country": "United-States"
    }
    
    # Sample data 2: Profile likely earning >50K
    data_high_income = {
        "age": 45,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }
    
    print("=" * 80)
    print("Census Income Prediction API - Live POST Test")
    print("=" * 80)
    
    # Test 1: Low income profile
    print("\n" + "=" * 80)
    print("\n[Test 1] POST Request - Low Income Profile")
    print("-" * 80)
    print(f"Input Data: {json.dumps(data_low_income, indent=2)}")
    status_code, result = post_to_api(api_predict, data_low_income)
    
    if status_code:
        print(f"\nStatus Code: {status_code}")
        print(f"Prediction Result: {json.dumps(result, indent=2)}")
    else:
        print("\nFailed to get response from API")
    
    # Test 2: High income profile
    print("\n" + "=" * 80)
    print("\n[Test 2] POST Request - High Income Profile")
    print("-" * 80)
    print(f"Input Data: {json.dumps(data_high_income, indent=2)}")
    status_code, result = post_to_api(api_predict, data_high_income)
    
    if status_code:
        print(f"\nStatus Code: {status_code}")
        print(f"Prediction Result: {json.dumps(result, indent=2)}")
    else:
        print("\nFailed to get response from API")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
