from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd   

pipe = joblib.load("model.joblib")
app = FastAPI(title="Bank Churn API", version="1.0")

class CustomerFeatures(BaseModel):
    credit_score: int
    country: str
    gender: str
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int
    active_member: int
    estimated_salary: float

@app.get("/")
def root():
    return {"status": "ok", "message": "Bank Churn API. See /docs"}

@app.post("/predict")
def predict(payload: CustomerFeatures):
    X = pd.DataFrame([payload.model_dump()])        
    proba = pipe.predict_proba(X)[0, 1]       
    label = int(proba >= 0.5)
    return {"churn_probability": float(proba), "churn_pred": label}