from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from schemas import (
    IrisFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse
)
from model import get_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="IRIS Prediction API",
    description="API for predicting IRIS flower species using a trained ML model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load the model at startup"""
    try:
        logger.info("🔄 Loading model on startup...")
        model = get_model()
        logger.info("✅ Model loaded successfully from %s", model.model_path)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the IRIS Prediction API 🚀",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if API and model are healthy"""
    try:
        model = get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=model.is_loaded(),
            model_path=model.model_path,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: IrisFeatures):
    """Predict the IRIS species for a single instance"""
    try:
        model = get_model()
        feature_list = [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]
        result = model.predict(feature_list)
        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Predict IRIS species for multiple instances"""
    try:
        model = get_model()
        features_list = [
            [inst.sepal_length, inst.sepal_width, inst.petal_length, inst.petal_width]
            for inst in request.instances
        ]
        results = model.predict_batch(features_list)
        predictions = [PredictionResponse(**r) for r in results]
        return BatchPredictionResponse(predictions=predictions, total=len(predictions))

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get metadata about the loaded model"""
    try:
        model = get_model()
        return {
            "model_path": model.model_path,
            "model_loaded": model.is_loaded(),
            "classes": model.class_names,
            "num_features": len(model.feature_names),
            "feature_names": model.feature_names
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
