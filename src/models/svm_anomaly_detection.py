# Auteur : Mohamed Amine ZAGRANI

from sklearn.svm import OneClassSVM

def train_anomaly_svm(data, features, kernel='rbf', nu=0.1):
    model = OneClassSVM(kernel=kernel, nu=nu)
    model.fit(data[features])
    return model

def detect_outliers_svm(model, data, features):
    predictions = model.predict(data[features])
    return predictions
