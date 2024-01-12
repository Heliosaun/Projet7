from flask import Flask, request, jsonify
import pandas as pd
import pickle
import shap

# Initialisation de l'application Flask
app = Flask(__name__)

# Fonction pour charger les données et le modèle
def load_data_and_model():
    data = pd.read_csv('data.csv', index_col='SK_ID_CURR', encoding='utf-8')
    sample = pd.read_csv('sample.csv', index_col='SK_ID_CURR', encoding='utf-8')
    clf = pickle.load(open('model_opti_lgbm_final.pkl', 'rb'))
    return data, sample, clf

# Fonction pour obtenir la décision de crédit
def get_credit_decision(score):
    decision = "CREDIT ACCORDE" if score < 0.5 else "CREDIT REFUSE"
    decision_color = "green" if score < 0.5 else "red"
    return decision, decision_color

# Route pour la prédiction de crédit
@app.route('/predict_credit', methods=['POST'])
def predict_credit():
    data, sample, clf = load_data_and_model()

    # Récupérer l'ID client de la requête
    chk_id = request.json['id_client']
    if chk_id is None or not str(chk_id).isdigit():
        return jsonify({"error": "Invalid client ID"}), 400
    X = sample.loc[[int(chk_id)]]
    score = clf.predict_proba(X.iloc[:, :-1])[:, 1][0]

    decision, decision_color = get_credit_decision(score)
    return jsonify({
        "id_client": chk_id,
        "decision": decision,
        "decision_color": decision_color,
        "score": score
    })

# Route pour les informations du client
@app.route('/client_info', methods=['GET'])
def client_info():
    data, sample, _ = load_data_and_model()
    chk_id = request.args.get('id_client')
    if chk_id is None or not chk_id.isdigit():
        return jsonify({"error": "Invalid client ID"}), 400
    client_info = data.loc[int(chk_id), ['CODE_GENDER', 'DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'NAME_FAMILY_STATUS']]
    client_info['Age'] = convert_days_to_years_months(client_info['DAYS_BIRTH'])
    client_info = client_info[['CODE_GENDER', 'Age', 'AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'NAME_FAMILY_STATUS']]

    return jsonify(client_info.to_dict())

# Fonction pour convertir les jours en années et mois
def convert_days_to_years_months(days):
    years = days // 365
    months = (days % 365) // 30  # approximation du nombre de mois
    return f"{years} ans, {months} mois"

# Démarrer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)

    # cd C:\Users\Heliosaun\Desktop\Projet7\FINAL\tests
    # python api.py
    # http://127.0.0.1:5000
