import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

def get_crypto_data(ticker='BTC-USD', period='90d', interval='1d'):
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    return data

def create_features(df):
    df['Return'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Close'].rolling(window=5).std()
    df.dropna(inplace=True)
    return df

def random_forest_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    print("\n Métricas Random Forest")
    print(f"Acurácia:  {acc:.4f}")
    print(f"Precisão:  {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nMatriz de Confusão:\n", conf_matrix)

 
    metrics_df = pd.DataFrame({
        "Métrica": ["Acurácia", "Precisão", "Recall", "F1-Score"],
        "Valor": [acc, precision, recall, f1]
    })
    metrics_df.to_csv("metricas_random_forest.csv", index=False)
    print("\n Métricas salvas em: metricas_random_forest.csv")

    return model


df = get_crypto_data()
df = create_features(df)

features = ['Return', 'MA5', 'MA10', 'Volatility']
X = df[features]
y = df['Target']

model = random_forest_model(X, y)
