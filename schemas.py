from pydantic import BaseModel, Field, validator
from typing import List

class IrisFeatures(BaseModel):
    """Input features for IRIS prediction"""
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=10, description="Petal width in cm")
    
    @validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    def check_positive(cls, v):
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str = Field(..., description="Predicted IRIS species")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: dict = Field(..., description="Probabilities for each class")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "setosa",
                "confidence": 0.95,
                "probabilities": {
                    "setosa": 0.95,
                    "versicolor": 0.03,
                    "virginica": 0.02
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    instances: List[IrisFeatures]
    
    class Config:
        json_schema_extra = {
            "example": {
                "instances": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2
                    },
                    {
                        "sepal_length": 6.7,
                        "sepal_width": 3.1,
                        "petal_length": 4.7,
                        "petal_width": 1.5
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    total: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_path: str
    version: str
