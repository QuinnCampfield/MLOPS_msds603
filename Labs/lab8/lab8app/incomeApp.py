from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
import numpy as np

app = FastAPI(
    title="Income prediction",
    description="Predict the income for an individual",
    version="0.1",
)

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'This is a model for predicting income'}

class IncomeFeatures(BaseModel):
    features: list[float]

@app.on_event('startup')
def load_artifacts():
    global model
    try:
        model = joblib.load("models/income_model.joblib")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None


# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data: IncomeFeatures):
    if model is None:
        return {'error': 'Model not loaded'}
    
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return {'predicted_income': float(prediction)}
    except Exception as e:
        return {'error': str(e)}