import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import List

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
    return "Hello world! Welcome to the Default Predictor API!"

# test local : http://127.0.0.1:8000/credit/
@app.get("/credit", response_model=ItemResponse)
def liste_identifiants():
    idx_client_list = data_test.index.tolist()
    return {
        "liste_features": liste_features,
        "idx_client": idx_client_list,
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
@app.get("/credit/{id_client}/shap", response_model=ShapValuesResponseModel)
def shap_values_client(id_client: int):
    if id_client in liste_id:
        idx = liste_id.index(id_client)
        data_client = data_test.iloc[idx:idx+1]
        shap_values = explainer.shap_values(data_client)
        shap_data_flat = [float(val) for val in shap_values[0].ravel()]
        return {"shap_val": shap_data_flat}
    else:
        raise HTTPException(status_code=404, detail="ID inconnu")

# to run the app in local : uvicorn app:app --reload
# to check the automatic interactive FastAPI documentation : http://127.0.0.1:8000/docs
