from pydantic import BaseModel, Field
from typing import Literal


# Pydantic model for census data input
# Using Field(alias=...) to handle column names with hyphens
class CensusData(BaseModel):
    age: int
    workclass: Literal[
        "State-gov", "Self-emp-not-inc", "Private", "Federal-gov",
        "Local-gov", "Self-emp-inc", "Without-pay", "Never-worked"
    ]
    fnlgt: int
    education: Literal[
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ]
    education_num: int = Field(alias="education-num")
    marital_status: Literal[
        "Never-married", "Married-civ-spouse", "Divorced",
        "Married-spouse-absent", "Separated", "Married-AF-spouse", "Widowed"
    ] = Field(alias="marital-status")
    occupation: Literal[
        "Adm-clerical", "Exec-managerial", "Handlers-cleaners",
        "Prof-specialty", "Other-service", "Sales", "Craft-repair",
        "Transport-moving", "Farming-fishing", "Machine-op-inspct",
        "Tech-support", "Protective-serv", "Armed-Forces", "Priv-house-serv"
    ]
    relationship: Literal[
        "Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"
    ]
    race: Literal["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
    sex: Literal["Male", "Female"]
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        # Allow population by field name or alias
        populate_by_name = True
        # JSON schema example for documentation
        json_schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }


# Pydantic model for prediction response
class PredictionResponse(BaseModel):
    prediction: str
