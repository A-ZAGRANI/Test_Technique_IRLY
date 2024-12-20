# Auteur : Mohamed Amine ZAGRANI

from sklearn.ensemble import IsolationForest

def train_anomaly_model(data, features):
    model = IsolationForest(n_estimators=100, random_state=42)
    model.fit(data[features])
    return model

def predict_anomalies(model, data, features):
    return model.predict(data[features])
