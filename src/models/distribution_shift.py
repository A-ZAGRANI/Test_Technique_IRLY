# Auteur : Mohamed Amine ZAGRANI

from sklearn.metrics import roc_auc_score

def calculate_shift_score(train_data, test_data, features):
    train_mean = train_data[features].mean()
    test_mean = test_data[features].mean()
    shift_scores = (test_mean - train_mean).abs()
    return shift_scores
