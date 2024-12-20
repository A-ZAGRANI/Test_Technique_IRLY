# Auteur : Mohamed Amine ZAGRANI

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath, test_size=0.2):
    data = pd.read_csv(filepath)
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, test_data
