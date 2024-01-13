import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import os
import plotly.express as px
import plotly.graph_objects as go
import requests
plt.style.use('fivethirtyeight')

def display_client_data(api_url, client_id):
    try:
        response = requests.get(f"{api_url}credit/{client_id}/data")
        response.raise_for_status()
        donnees_client = response.json()
        if 'data' in donnees_client:
            st.write("Données du Client :", donnees_client['data'])
        else:
            st.error("Les données du client sont manquantes dans la réponse de l'API.")
    except Exception as e:
        st.error(f"Erreur lors de l'affichage des données du client : {e}")

def display_credit_probability(api_url, client_id):
    try:
        response = requests.get(f"{api_url}credit/{client_id}/predict")
        response.raise_for_status()
        prediction = response.json()
        probability = prediction['probability']
        st.write("Probabilité de Score de Crédit :", probability)
        return probability
    except Exception as e:
        st.error(f"Erreur lors de l'affichage de la probabilité de score de crédit : {e}")
        return None

def display_summary_table(api_url, client_id):
    try:
        response = requests.get(f"{api_url}credit/{client_id}/data")
        response.raise_for_status()
        data = response.json()['data']
        df = pd.DataFrame(data)
        df['DAYS_BIRTH'] = df['DAYS_BIRTH'].apply(lambda x: f"{-x // 365} ans, {-x % 365 // 30} mois")
        summary = df[['CODE_GENDER', 'DAYS_BIRTH', 'AMT_INCOME_TOTAL']]
        st.table(summary)
    except Exception as e:
        st.error(f"Erreur lors de l'affichage des données du client : {e}")

def display_probability_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability,
        title = {'text': "Probabilité de Défaut (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "blue"},
            'steps': [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 50], 'color': 'yellow'},
                {'range': [50, 100], 'color': 'red'}
            ],
        }
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig)

def display_shap_values(api_url, client_id):
    try:
        response = requests.get(f"{api_url}credit/{client_id}/shap")
        response.raise_for_status()
        valeurs_shap = response.json()

        # Vérifier la structure des valeurs SHAP
        shap_values = valeurs_shap['shap_val']
        feature_names = valeurs_shap['feature_names']

        # Transformer les valeurs SHAP en array NumPy si elles ne sont pas déjà dans ce format
        shap_values_array = np.array(shap_values)

        # S'assurer que la longueur des valeurs SHAP correspond au nombre de caractéristiques
        if shap_values_array.shape[1] != len(feature_names):
            raise ValueError("La longueur des valeurs SHAP ne correspond pas au nombre de caractéristiques")

        # Création du DataFrame pour le graphique
        df_shap = pd.DataFrame(shap_values_array, columns=feature_names).T
        df_shap.columns = ['SHAP Value']
        df_shap = df_shap.sort_values(by='SHAP Value', ascending=False).head(10)

        # Création et affichage du graphique
        plt.figure(figsize=(10, 5))
        plt.barh(df_shap.index, df_shap['SHAP Value'])
        plt.xlabel('Importance SHAP')
        plt.title('Top 10 des caractéristiques les plus importantes')
        st.pyplot(plt.gcf())
    except Exception as e:
        st.error(f"Erreur lors de l'affichage des valeurs SHAP : {e}")

def main():
    st.title("Dashboard Scoring Credit")
    api_url = "https://p7-app-068fe8868110.herokuapp.com/"
    
    try:
        response = requests.get(f"{api_url}credit")
        response.raise_for_status()
        data = response.json()

        if 'idx_client' in data:
            sk_id_curr_list = data['idx_client']
            sk_id_choisi = st.selectbox("Choisir l'ID du Client", sk_id_curr_list)
            
            probability = display_credit_probability(api_url, sk_id_choisi)
            if probability is not None:
                decision = "CREDIT ACCORDE" if probability <= 30 else "CREDIT REFUSE"
                decision_color = "green" if probability <= 30 else "red"
                st.markdown(f"<h2 style='color: {decision_color}; text-align: center;'>{decision}</h2>", unsafe_allow_html=True)

                display_probability_gauge(probability)
                display_summary_table(api_url, sk_id_choisi)
                display_shap_values(api_url, sk_id_choisi)
        else:
            st.error("La clé 'idx_client' est manquante dans la réponse de l'API.")
    except Exception as e:
        st.error(f"Une erreur s'est produite : {e}")

    
if __name__ == "__main__":
    main()
    
        # TEST LOCAL
    # cd C:\Users\Heliosaun\Desktop\Projet7\FINALTEST
    # python -m streamlit run dashboard.py
    # Network URL: http://192.168.1.173:8501
