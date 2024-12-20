### Auteur : Mohamed Amine ZAGRANI


# Détection des Anomalies et Scoring de Dérive

Ce projet implémente un système de détection des anomalies et de scoring de dérive basé sur le jeu de données Amazon Reviews Dataset.

## Structure du Projet

- `data/` : Contient les données utilisées pour l'entraînement et le test.
- `notebooks/` : Contient les notebooks pour l'exploration des données.
- `src/` : Contient le code source du projet.
    - `models/` : Implémentation des modèles de machine learning.
    - `preprocessing/` : Code pour le prétraitement des données.
    - `api/` : Déploiement de l'API.
- `tests/` : Tests unitaires pour valider le code.
- `requirements.txt` : Liste des dépendances.

## Instructions

1. Installer les dépendances :
    ```
    pip install -r requirements.txt
    ```

2. Exécuter l'API :
    ```
    uvicorn src.api.main:app --reload
    ```

3. Suivre les notebooks pour explorer les données.

## Auteur

Ce projet est développé dans le cadre d'un test technique.
