# test_api.py
from flask_testing import TestCase
from api import app  # Import de l'application Flask
import json

class TestFlaskApi(TestCase):
    def create_app(self):
        # Création d'une instance de l'application pour les tests
        app.config['TESTING'] = True
        return app

    def test_predict_credit(self):
        # Test de la route predict_credit avec un ID client valide
        response = self.client.post('/predict_credit', 
                                    data=json.dumps({'id_client': 340510}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        # Autres assertions

    def test_client_info(self):
        # Test de la route client_info avec un ID client valide
        response = self.client.get('/client_info?id_client=340510')
        self.assertEqual(response.status_code, 200)
        # Autres assertions

    def test_predict_credit_invalid_id(self):
        # Test de la route predict_credit avec un ID client invalide
        response = self.client.post('/predict_credit', 
                                    data=json.dumps({'id_client': 'invalide'}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)  # API renvoyant un statut 400 pour une entrée invalide
        # Vérifie également le message d'erreur renvoyé par l'API

    def test_client_info_no_id(self):
        # Test de la route client_info sans ID client
        response = self.client.get('/client_info')
        self.assertEqual(response.status_code, 400)  # API renvoyant un statut 400 pour une requête sans ID client
        # Autres assertions
        
    def test_predict_credit_score_range(self):
        # Test pour vérifier que le score de prédiction est entre 0 et 1
        response = self.client.post('/predict_credit', data=json.dumps({'id_client': 340510}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = response.json
        predicted_score = data['score']
    
        # Vérifie que le score est compris entre 0 et 1
        self.assertGreaterEqual(predicted_score, 0, "Le score de prédiction devrait être au moins 0")
        self.assertLessEqual(predicted_score, 1, "Le score de prédiction devrait être au plus 1")




    
# cd C:\Users\Heliosaun\Desktop\Projet7\FINAL\tests
