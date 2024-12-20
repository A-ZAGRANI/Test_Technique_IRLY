# Auteur : Mohamed Amine ZAGRANI
import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import json

# Ajout du chemin racine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.anomaly_detection import train_anomaly_model, predict_anomalies
from src.models.distribution_shift import calculate_shift_score

app = FastAPI()

# Classe pour le format d'entrée
class DataInput(BaseModel):
    data: list

# Route racine pour l'accueil
@app.get("/")
def root():
    return {"message": "Bienvenue dans l'API. Consultez /docs pour voir la documentation."}

# Route pour détecter les anomalies
@app.post("/anomaly-detection/")
def detect_anomalies(input_data: DataInput):
    data = pd.DataFrame(input_data.data)
    model = train_anomaly_model(data, features=["feature1", "feature2"])  # Ajouter les features selon vos données
    predictions = predict_anomalies(model, data, features=["feature1", "feature2"])
    return {"anomalies": predictions.tolist()}

# Route pour détecter les décalages de distribution
@app.post("/distribution-shift/")
def distribution_shift(input_data: DataInput):
    data = pd.DataFrame(input_data.data)
    shift_scores = calculate_shift_score(data[:50], data[50:], features=["feature1", "feature2"])  # Placeholder
    return {"shift_scores": shift_scores.tolist()}

# Lister tous les notebooks disponibles
@app.get("/notebooks/")
def list_notebooks():
    notebook_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../notebooks'))
    try:
        notebooks = [f for f in os.listdir(notebook_dir) if f.endswith(".ipynb")]
        return {"notebooks": notebooks}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Le répertoire des notebooks est introuvable.")

# Lire un notebook spécifique
@app.get("/notebooks/{notebook_name}")
def get_notebook(notebook_name: str):
    notebook_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../notebooks'))
    notebook_path = os.path.join(notebook_dir, notebook_name)
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"notebook_name": notebook_name, "content": content}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_name} introuvable.")

# Exécuter un notebook
@app.post("/run-notebook/{notebook_name}")
def run_notebook(notebook_name: str):
    notebook_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../notebooks'))
    notebook_path = os.path.join(notebook_dir, notebook_name)
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        executor = ExecutePreprocessor(timeout=600, kernel_name='python3')
        executor.preprocess(nb, {'metadata': {'path': notebook_dir}})

        executed_notebook_path = os.path.join(notebook_dir, f"executed_{notebook_name}")
        with open(executed_notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return {"message": f"Notebook {notebook_name} exécuté avec succès.", "output_path": executed_notebook_path}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_name} introuvable.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'exécution du notebook: {str(e)}")
