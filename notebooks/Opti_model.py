import optuna
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.lightgbm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Charger les données
df = pd.read_csv("C:/Users/Heliosaun/Desktop/Projet7/final_dataset.csv"")
feats = [f for f in df.columns if f not in ['TARGET', 'SK_ID_CURR']]

# Séparation des données
X_train, X_val, y_train, y_val = train_test_split(df[feats], df['TARGET'], test_size=0.2, random_state=42)

# Remplacer les valeurs `NaN` par la médiane de chaque colonne
X_train.fillna(X_train.median(), inplace=True)
X_val.fillna(X_val.median(), inplace=True)
y_train.fillna(y_train.median(), inplace=True)
y_val.fillna(y_val.median(), inplace=True)

# Remplacer les caractères spéciaux dans les noms des colonnes
df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
X_train.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]
X_val.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X_val.columns]

# Créer une nouvelle expérience MLflow
mlflow.create_experiment(name='hyper_opti_tuning_final')
mlflow.set_experiment('hyper_opti_tuning_final')

# Fonction coût personnalisée
def custom_cost_function(y_true, y_pred, false_negative_weight=10):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp + false_negative_weight * fn

# Fonction objectif pour Optuna
def objective(trial):
    with mlflow.start_run(run_name='LightGBM_Optuna_Trial_FINAL'):
        params = {
            'nthread': trial.suggest_int('nthread', 1, 16),
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'n_estimators': trial.suggest_int('n_estimators', 200, 10000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'max_depth': trial.suggest_int('max_depth', 1, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.001, 0.1),
            'min_child_weight': trial.suggest_float('min_child_weight', 5, 50)
        }

        callbacks = [early_stopping(stopping_rounds=200)]
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)

        y_pred_val = clf.predict(X_val)
        y_pred_proba = clf.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, y_pred_proba)
        custom_cost = custom_cost_function(y_val, y_pred_val)
        
        # Log metrics and parameters in MLflow
        mlflow.log_params(params)
        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("Custom_Cost", custom_cost)
        
        # Feature Importance plot
        feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, feats)), columns=['Value','Feature'])
        plt.figure(figsize=(20, 10))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
        plt.title('LightGBM Features Importance')
        plt.tight_layout()
        plt.savefig("C:/Users/Heliosaun/Desktop/Projet7/feature_importance.png")
        mlflow.log_artifact("C:/Users/Heliosaun/Desktop/Projet7/FINAL/docs/feature_importance.png")

        # Confusion Matrix plot
        conf_mat = confusion_matrix(y_val, y_pred_val)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("C:/Users/Heliosaun/Desktop/Projet7/FINAL/docs/confusion_matrix.png")
        mlflow.log_artifact("C:/Users/Heliosaun/Desktop/Projet7/FINAL/docs/confusion_matrix.png")

        # ROC Curve plot
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {auc})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig("C:/Users/Heliosaun/Desktop/Projet7/FINAL/docs/roc_curve.png")
        mlflow.log_artifact("C:/Users/Heliosaun/Desktop/Projet7/FINAL/docs/roc_curve.png")

        # Sauvegarder le modèle
        mlflow.lightgbm.log_model(clf, "LGBM_opti")

        return custom_cost, auc

# Initialiser une étude Optuna
study = optuna.multi_objective.create_study(directions=['minimize', 'maximize'])
study.optimize(objective, n_trials=30)

# Récupérer et enregistrer le meilleur modèle
best_trials = study.get_pareto_front_trials()
best_trial = best_trials[0]
best_params = best_trial.params

with mlflow.start_run(run_name='Best_Optuna_Results'):
    mlflow.log_params(best_params)
    mlflow.log_metric("Best_Custom_Cost", best_trial.values[0])
    mlflow.log_metric("Best_AUC", best_trial.values[1])