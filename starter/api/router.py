from fastapi import APIRouter, status, HTTPException
from api.utils import CensusData, PredictionResponse
import pickle
import pandas as pd
import os
import sys

# Add the parent directory to the path to import from starter module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from starter.ml.data import process_data
from starter.ml.model import inference

router = APIRouter()

# Load the model and artifacts at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model")

with open(os.path.join(MODEL_PATH, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_PATH, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)
with open(os.path.join(MODEL_PATH, "lb.pkl"), "rb") as f:
    lb = pickle.load(f)

# Define categorical features
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@router.get("/", status_code=status.HTTP_200_OK)
async def root() -> dict:
    """
    Welcome message for the API root endpoint.
    
    Returns:
        dict: A welcome message
    """
    return {"message": "Welcome to the Census Income Prediction API!"}


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(data: CensusData) -> PredictionResponse:
    """
    Perform model inference on census data.
    
    Args:
        data: Census data input conforming to CensusData model
        
    Returns:
        PredictionResponse: Prediction result
        
    Raises:
        HTTPException: If an error occurs during prediction
    """
    try:
        # Convert Pydantic model to dictionary and then to DataFrame
        data_dict = {
            "age": data.age,
            "workclass": data.workclass,
            "fnlgt": data.fnlgt,
            "education": data.education,
            "education-num": data.education_num,
            "marital-status": data.marital_status,
            "occupation": data.occupation,
            "relationship": data.relationship,
            "race": data.race,
            "sex": data.sex,
            "capital-gain": data.capital_gain,
            "capital-loss": data.capital_loss,
            "hours-per-week": data.hours_per_week,
            "native-country": data.native_country,
        }
        
        # Create DataFrame with a single row
        df = pd.DataFrame([data_dict])
        
        # Process the data (no label for inference)
        X, _, _, _ = process_data(
            df,
            categorical_features=CAT_FEATURES,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb
        )
        
        # Make prediction
        pred = inference(model, X)
        
        # Convert prediction back to label
        prediction_label = lb.inverse_transform(pred)[0]
        
        return PredictionResponse(prediction=prediction_label)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
