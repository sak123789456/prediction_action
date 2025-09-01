# Fichier: model_trainer.py (Version avec Métriques de Régression)
# Description: Gère l'entraînement, l'évaluation et la prédiction des modèles.

import pandas as pd
import ta
import numpy as np # NOUVEAU : Import pour le calcul de la racine carrée (RMSE)

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# NOUVEAU : Importations pour les métriques de régression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, r2_score, mean_squared_error

def create_features(df):
    """Crée des variables (features) à partir des données de prix."""
    # (Cette fonction ne change pas)
    df = df.copy()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Signal'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])
    stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['Stochastic'] = stoch.stoch()
    df.dropna(inplace=True)
    return df

def train_evaluate_and_predict(data):
    """
    Entraîne, évalue les modèles et retourne les prédictions et les métriques.
    """
    # 1. Préparation des données
    df = create_features(data)
    df['Target_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['Target_Price'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_30', 
        'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'BB_Signal', 'Stochastic'
    ]
    
    X = df[features]
    X_predict = X.iloc[-1:].copy()
    X_train_full = X.iloc[:-1]
    y_class_full = df['Target_Direction'].iloc[:-1]
    y_reg_full = df['Target_Price'].iloc[:-1]

    # 2. Mise à l'échelle des features
    print("🔧 Mise à l'échelle des données...")
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_predict_scaled = scaler.transform(X_predict)

    # 3. Séparation des données pour l'ÉVALUATION des deux modèles
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X_train_full_scaled, y_class_full, y_reg_full, test_size=0.2, random_state=42
    )

    # 4. ÉVALUATION DU MODÈLE DE CLASSIFICATION
    print("📊 Évaluation du modèle de classification (MLP)...")
    clf_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    clf_model.fit(X_train, y_class_train)
    y_pred_class = clf_model.predict(X_test)
    y_pred_proba = clf_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_class_test, y_pred_class),
        "precision": precision_score(y_class_test, y_pred_class, zero_division=0),
        "recall": recall_score(y_class_test, y_pred_class, zero_division=0),
        "roc_auc": roc_auc_score(y_class_test, y_pred_proba)
    }

    # 5. NOUVEAU : ÉVALUATION DU MODÈLE DE RÉGRESSION
    print("📊 Évaluation du modèle de régression (MLP)...")
    reg_model_eval = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    reg_model_eval.fit(X_train, y_reg_train)
    y_pred_reg = reg_model_eval.predict(X_test)

    # Ajout des métriques de régression au dictionnaire
    metrics["r2"] = r2_score(y_reg_test, y_pred_reg)
    metrics["rmse"] = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
    
    # Affichage de toutes les métriques dans le terminal
    for key, value in metrics.items():
        print(f"  - {key.upper()}: {value:.4f}")

    # 6. ENTRAÎNEMENT FINAL ET PRÉDICTION
    print("🧠 Entraînement final des modèles...")
    clf_model.fit(X_train_full_scaled, y_class_full)
    direction_pred_code = clf_model.predict(X_predict_scaled)[0]
    direction_pred_label = "HAUSSE" if direction_pred_code == 1 else "BAISSE"
    
    reg_model_final = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    reg_model_final.fit(X_train_full_scaled, y_reg_full)
    price_pred = reg_model_final.predict(X_predict_scaled)[0]

    return direction_pred_label, float(price_pred), metrics