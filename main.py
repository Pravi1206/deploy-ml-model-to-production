from fastapi import FastAPI
from api.router import router

# Initialize FastAPI app
app = FastAPI(
    title="Census Income Prediction API",
    description="API for predicting income levels based on census data",
    version="1.0.0"
)

# Include the router
app.include_router(router)
