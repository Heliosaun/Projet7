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
    
def test_get_age_distribution():
    # Remplacez 430080 par un ID client valide de votre dataset
    response = client.get("/credit/430080/age")
    assert response.status_code == 200
    assert "age_distribution" in response.json()
    # Vous pouvez ajouter des assertions supplémentaires pour tester les valeurs spécifiques si nécessaire

def test_get_income_distribution():
    response = client.get("/distributions/income")
    assert response.status_code == 200
    assert "income_distribution" in response.json()
