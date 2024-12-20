import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.preprocessing.data_cleaning import preprocess_data
from src.models.svm_anomaly_detection import train_anomaly_svm, detect_outliers_svm
from src.models.kl_shift_scoring import compute_kl_divergence


def test_preprocess_data():
    # Chargement du sous-ensemble
    data = pd.read_csv("project/data/amazon_reviews_real_subset.csv")
    processed_data = preprocess_data(filepath="project/data/amazon_reviews_real_subset.csv",
                                     text_column="text",
                                     numeric_columns=["rating"])
    # Vérification que les colonnes essentielles sont intactes
    assert "text" in processed_data.columns
    assert processed_data["rating"].isnull().sum() == 0  # Pas de valeurs manquantes après traitement

def test_anomaly_detection():
    # Chargement du sous-ensemble
    data = pd.read_csv("project/data/amazon_reviews_real_subset.csv")
    features = ["rating"]
    model = train_anomaly_svm(data, features)
    predictions = detect_outliers_svm(model, data, features)
    # Vérification de la présence de résultats
    assert len(predictions) == len(data)

def test_kl_divergence():
    # Chargement du sous-ensemble
    data = pd.read_csv("project/data/amazon_reviews_real_subset.csv")
    train_data = data.iloc[:500]  # Utilisation de 500 premiers exemples pour entraîner
    test_data = data.iloc[500:]   # Utilisation du reste pour tester
    score = compute_kl_divergence(train_data, test_data, "rating")
    # Vérification que le score est calculé
    assert score >= 0
