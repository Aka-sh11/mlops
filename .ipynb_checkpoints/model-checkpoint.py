import joblib
import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IrisModel:
    """Wrapper class for IRIS model"""

    def __init__(self, model_path: str = "models/iris_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.class_names = ["setosa", "versicolor", "virginica"]
        self.feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        self.load_model()

    def load_model(self):
        """Load model from disk"""
        if not os.path.exists(self.model_path):
            msg = f"Model file not found at {self.model_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, features: list) -> dict:
        """Make prediction for a single sample"""
        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            X = pd.DataFrame([features], columns=self.feature_names)
            pred_idx = self.model.predict(X)[0]

            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(X)[0]
            else:
                probas = np.zeros(len(self.class_names))
                probas[pred_idx] = 1.0

            confidence = float(np.max(probas))
            prob_dict = {cls: float(p) for cls, p in zip(self.class_names, probas)}

            return {
                "prediction": self.class_names[pred_idx],
                "confidence": confidence,
                "probabilities": prob_dict
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def predict_batch(self, features_list: list) -> list:
        """Make predictions for multiple samples"""
        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            X = pd.DataFrame(features_list, columns=self.feature_names)
            preds = self.model.predict(X)

            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(X)
            else:
                probas = np.eye(len(self.class_names))[preds]

            results = []
            for i, pred_idx in enumerate(preds):
                prob_dict = {cls: float(p) for cls, p in zip(self.class_names, probas[i])}
                results.append({
                    "prediction": self.class_names[pred_idx],
                    "confidence": float(np.max(probas[i])),
                    "probabilities": prob_dict
                })
            return results

        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise

    def is_loaded(self) -> bool:
        return self.model is not None


# Global singleton
_model_instance = None


def get_model() -> IrisModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = IrisModel()
    return _model_instance
