from fastapi import APIRouter
from api.utils import CensusData, PredictionResponse

router = APIRouter()


@router.get("/")
async def root() -> dict:
    """
    Welcome message for the API root endpoint.
    
    Returns:
        dict: A welcome message
    """
    return {"message": "Welcome to the Census Income Prediction API!"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(data: CensusData) -> PredictionResponse:
    """
    Perform model inference on census data.
    
    Args:
        data: Census data input conforming to CensusData model
        
    Returns:
        PredictionResponse: Prediction result
    """
    # TODO: Load your trained model and perform inference
    # For now, returning a placeholder prediction
    
    # Example of accessing the data with Python-friendly variable names:
    # data.education_num, data.marital_status, data.capital_gain, etc.
    
    # Placeholder prediction logic
    prediction = "<=50K"  # Replace with actual model inference
    
    return PredictionResponse(prediction=prediction)
