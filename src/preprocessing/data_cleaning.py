# Auteur : Mohamed Amine ZAGRANI

import pandas as pd
import numpy as np

def preprocess_data(filepath, text_column, numeric_columns):
    data = pd.read_csv(filepath)

    data[text_column] = data[text_column].fillna("").str.lower().str.replace("[^a-zA-Z0-9 ]", "")

    for col in numeric_columns:
        data[col] = (data[col] - data[col].mean()) / data[col].std()

    return data
