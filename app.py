import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import List
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import os

app = FastAPI()

# Définition des modèles Pydantic pour les réponses
class ListResponseModel(BaseModel):
    liste_features: List[str]

class ProbabilityResponseModel(BaseModel):
    probability: float

class ItemResponse(BaseModel):
    liste_features: List[str]
    idx_client: List[int]

class DataResponseModel(BaseModel):
    data: List[dict]

class ShapValuesResponseModel(BaseModel):
    shap_val: List[float]
    
class AgeDistributionResponseModel(BaseModel):
    age_distribution: dict

class IncomeDistributionResponseModel(BaseModel):
    income_distribution: dict

# Chargement et prétraitement des données
data_test = pd.read_csv("sample_final_dataset.csv")
liste_id = data_test["SK_ID_CURR"].tolist()  # Conserver les IDs clients
data_test = data_test.drop(columns=["SK_ID_CURR", "TARGET"], errors='ignore')
liste_features = data_test.columns.tolist()

# Chargement du modèle et initialisation de l'explainer SHAP
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
explainer = shap.TreeExplainer(model)

# test local : http://127.0.0.1:8000/
@app.get("/")
def welcome():
    return "Bienvenue dans l'API de scoring crédit!"

# test local : http://127.0.0.1:8000/credit/
@app.get("/credit", response_model=ItemResponse)
def liste_identifiants():
    return {
        "liste_features": liste_features,
        "idx_client": liste_id,
    }


# test local : http://127.0.0.1:8000/credit/430080/predict 
@app.get("/credit/{id_client}/predict", response_model=ProbabilityResponseModel)
def predict_score_client(id_client: int):
    if id_client in liste_id:
        idx = liste_id.index(id_client)
        data_client = data_test.iloc[idx:idx+1]  # Sélectionner les données du client
        proba = model.predict_proba(data_client)[:, 1]
        proba_0 = round(proba[0] * 100, 2)
        return {"probability": proba_0}
    else:
        raise HTTPException(status_code=404, detail="ID inconnu")

# test local : http://127.0.0.1:8000/credit/430080/data
@app.get("/credit/{id_client}/data", response_model=DataResponseModel)
def donnees_client(id_client: int):
    if id_client in liste_id:
        idx = liste_id.index(id_client)
        data_client = data_test.iloc[idx:idx+1]
        # Traitement des valeurs NaN ou extrêmes
        data_client = data_client.applymap(lambda x: None if pd.isna(x) else x)
        return {"data": data_client.to_dict(orient="records")}
    else:
        raise HTTPException(status_code=404, detail="ID inconnu")

# test local : http://127.0.0.1:8000/credit/430080/shap
@app.get("/credit/{id_client}/shap")
def shap_values_client(id_client: int):
    if id_client in liste_id:
        idx = liste_id.index(id_client)
        data_client = data_test.iloc[idx:idx+1]
        
        feature_names = data_test.columns[1:].tolist()
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_client)
        
        # Renvoyer les valeurs SHAP et les noms des caractéristiques
        return {
            "shap_val": shap_values[0].tolist(),
            "feature_names": data_test.columns.tolist()
        }
    else:
        raise HTTPException(status_code=404, detail="ID inconnu")
        
def calculate_age_in_years_and_months(days):
    years = -days // 365
    months = (-days % 365) // 30  # Approximation du nombre de mois
    return f"{years} ans, {months} mois"

# test local : http://127.0.0.1:8000/credit/430080/age
@app.get("/credit/{id_client}/age", response_model=AgeDistributionResponseModel)
def get_age_distribution(id_client: int):
    if id_client in liste_id:
        client_index = liste_id.index(id_client)
        client_days_birth = data_test.iloc[client_index]['DAYS_BIRTH']
        client_age = calculate_age_in_years_and_months(client_days_birth)
        
        # Calcul de la distribution de l'âge pour tous les clients
        age_distribution = data_test['DAYS_BIRTH'].apply(calculate_age_in_years_and_months).value_counts().to_dict()

        # Ajouter l'âge du client sélectionné à la réponse
        age_distribution['client_age'] = client_age
        return {"age_distribution": age_distribution}
    else:
        raise HTTPException(status_code=404, detail="ID client inconnu")

# test local : http://127.0.0.1:8000/distributions/income
@app.get("/distributions/income", response_model=IncomeDistributionResponseModel)
def get_income_distribution():
    income_distribution = data_test[data_test['AMT_INCOME_TOTAL'] < 2000000]['AMT_INCOME_TOTAL'].value_counts().to_dict()
    return {"income_distribution": income_distribution}


# to run the app in local : uvicorn app:app --reload
# to check the automatic interactive FastAPI documentation : http://127.0.0.1:8000/docs
