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

plt.style.use('fivethirtyeight')

def main():
    data, sample, clf = load_data_and_model()
    id_client = sample.index.values

    st.title("Application d'octroi de crédit")
    chk_id = st.selectbox("ID Client", id_client)

    prediction = load_prediction(sample, chk_id, clf)
    decision, decision_color = get_credit_decision(prediction)
    st.markdown(f'<h2 style="text-align: center; color: {decision_color};">{decision}</h2>', unsafe_allow_html=True)
    display_probability_gauge(prediction)

    display_client_info(data, chk_id)  # Informations client

    display_age_distribution(data, chk_id)  # Graphiques de distribution
    display_income_distribution(data, chk_id)
    
    display_feature_importance(sample, chk_id, clf) # SHAP: top10

@st.cache_data
def load_data_and_model():
    data = pd.read_csv('data.csv', index_col='SK_ID_CURR', encoding='utf-8')
    sample = pd.read_csv('sample.csv', index_col='SK_ID_CURR', encoding='utf-8')
    clf = pickle.load(open('model_opti_lgbm_final.pkl', 'rb'))
    return data, sample, clf

def display_client_solvability(sample, chk_id, clf):
    score = clf.predict_proba(sample[sample.index == int(chk_id)].iloc[:, :-1])[0, 1]
    decision = "CREDIT ACCORDE" if score < 0.5 else "CREDIT REFUSE"
    decision_color = "green" if score < 0.5 else "red"
    
    # Affichage de la décision en gros caractères
    st.markdown(f'<h2 style="color: {decision_color};">{decision}</h2>', unsafe_allow_html=True)

    # Affichage de la jauge Plotly
    display_probability_gauge(score)

def display_probability_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de défaut (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "lightgray"},
            'steps': [
                {'range': [0, 40], 'color': 'green'},
                {'range': [40, 50], 'color': 'yellow'},
                {'range': [50, 100], 'color': 'red'}
            ],
        }
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig)

def display_age_distribution(data, chk_id):
    infos_client = data.loc[int(chk_id)]
    client_age = infos_client['DAYS_BIRTH'] / 365
    data_age = data["DAYS_BIRTH"] / 365

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data_age, edgecolor='k', color="blue", bins=20)
    ax.axvline(client_age, color="red", linestyle='--')
    ax.set(title='Distribution des Ages', xlabel='Age (Années)', ylabel='Nombre')
    st.pyplot(fig)

def display_income_distribution(data, chk_id):
    infos_client = data.loc[int(chk_id)]  # Obtenir les informations du client
    client_income = infos_client['AMT_INCOME_TOTAL']
    data_income = data[data["AMT_INCOME_TOTAL"] < 200000]["AMT_INCOME_TOTAL"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data_income, edgecolor='k', color="blue", bins=10)
    ax.axvline(client_income, color="red", linestyle='--')
    ax.set(title='Distribution des Revenus', xlabel='Revenus (USD)', ylabel='Nombre')
    st.pyplot(fig)   
    
def display_client_info(data, chk_id):
    client_info = data.loc[int(chk_id), ['CODE_GENDER', 'DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'NAME_FAMILY_STATUS']]
    client_info['Age'] = convert_days_to_years_months(client_info['DAYS_BIRTH'])
    client_info = client_info[['CODE_GENDER', 'Age', 'AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'NAME_FAMILY_STATUS']]
    st.write(client_info.to_frame().T)

def convert_days_to_years_months(days):
    years = days // 365
    months = (days % 365) // 30  # approximation du nombre de mois
    return f"{years} ans, {months} mois"

def display_feature_importance(sample, chk_id, clf):
    shap.initjs()
    X = sample.iloc[:, :-1]
    X = X[X.index == chk_id]

    st.subheader("Top 10 des caractéristiques les plus importantes")
    fig, ax = plt.subplots(figsize=(10, 10))
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values[0], X, plot_type="bar", max_display=10, color_bar=False, plot_size=(5, 5))
    st.pyplot(fig)
        
def get_credit_decision(score):
    decision = "CREDIT ACCORDE" if score < 0.5 else "CREDIT REFUSE"
    decision_color = "green" if score < 0.5 else "red"
    return decision, decision_color

@st.cache_data()
def load_prediction(sample, chk_id, _clf):
    X = sample.loc[[int(chk_id)]]
    score = _clf.predict_proba(X.iloc[:, :-1])[:, 1]
    return score[0]

if __name__ == '__main__':
    main()
    
    # TEST LOCAL
    # cd C:\Users\Heliosaun\Desktop\Projet7\FINAL\src\dashboard
    # python -m streamlit run app.py
    # Network URL: http://192.168.1.173:8501
