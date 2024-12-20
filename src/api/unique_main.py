# Auteur : Mohamed Amine ZAGRANI
import sys
import os

# Ajoutez le chemin racine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.models.svm_anomaly_detection import train_anomaly_svm, detect_outliers_svm
from src.models.kl_shift_scoring import compute_kl_divergence

app = FastAPI()

class AnomalyInput(BaseModel):
    data: list
    features: list

@app.post("/anomaly-detection/")
def anomaly_detection(input_data: AnomalyInput):
    try:
        df = pd.DataFrame(input_data.data)
        model = train_anomaly_svm(df, input_data.features)
        predictions = detect_outliers_svm(model, df, input_data.features)
        return {"anomalies": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ShiftInput(BaseModel):
    train_data: list
    test_data: list
    feature: str

@app.post("/distribution-shift/")
def distribution_shift(input_data: ShiftInput):
    try:
        train_df = pd.DataFrame(input_data.train_data)
        test_df = pd.DataFrame(input_data.test_data)
        shift_score = compute_kl_divergence(train_df, test_df, input_data.feature)
        return {"shift_score": shift_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
