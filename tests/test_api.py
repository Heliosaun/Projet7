from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_welcome():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Bienvenue dans l'API de scoring crédit!"

def test_liste_identifiants():
    response = client.get("/credit")
    assert response.status_code == 200
    assert "liste_features" in response.json()
    assert "idx_client" in response.json()

def test_predict_score_client():
    # Remplacez 430080 par un ID client valide de votre dataset
    response = client.get("/credit/430080/predict")
    assert response.status_code == 200
    assert "probability" in response.json()

def test_donnees_client():
    # Remplacez 430080 par un ID client valide de votre dataset
    response = client.get("/credit/430080/data")
    assert response.status_code == 200
    assert "data" in response.json()

def test_shap_values_client():
    # Remplacez 430080 par un ID client valide de votre dataset
    response = client.get("/credit/430080/shap")
    assert response.status_code == 200
    assert "shap_val" in response.json()
