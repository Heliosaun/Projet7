from fastapi.testclient import TestClient
from app import app  # Ajustez l'import selon la structure de votre projet

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello world! Welcome to the Default Predictor API!"

def test_liste_identifiants():
    response = client.get("/credit")
    assert response.status_code == 200
    data = response.json()
    assert "liste_id" in data
    assert "liste_features" in data
    assert "idx_client" in data
    # Vous pouvez ajouter plus d'assertions ici pour vérifier le contenu spécifique de la réponse

def test_predict_score_client():
    test_id = 340510  # Remplacez par un ID client valide de votre jeu de données
    response = client.get(f"/credit/{test_id}/predict")
    assert response.status_code == 200
    assert "probability" in response.json()
    
# cd C:\Users\Heliosaun\Desktop\Projet7\FINAL\tests