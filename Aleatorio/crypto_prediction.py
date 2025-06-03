import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

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

def random_baseline(y_true):
    np.random.seed(42)
    y_pred = np.random.randint(0, 2, size=len(y_true))
    acc = accuracy_score(y_true, y_pred)
    return acc

def random_forest_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"\n Métricas do Random Forest:")
    print(f"Acurácia:  {acc:.4f}")
    print(f"Precisão:  {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nMatriz de Confusão:\n{conf_matrix}")
    print(f"\nRelatório de Classificação:\n{class_report}")

    return acc


df = get_crypto_data()
df = create_features(df)


features = ['Return', 'MA5', 'MA10', 'Volatility']
X = df[features]
y = df['Target']


baseline_acc = random_baseline(y)
print(f"\nBaseline Aleatória (cara ou coroa): {baseline_acc:.4f}")

rf_acc = random_forest_model(X, y)

#explicação feita no doc apresentado junto com a entrega do trabalho