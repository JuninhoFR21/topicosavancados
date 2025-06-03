import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


np.random.seed(42)
precos = np.sin(np.linspace(0, 50, 500)) + np.random.normal(0, 0.5, 500)
precos = precos.reshape(-1, 1)


scaler = MinMaxScaler(feature_range=(0, 1))
precos_norm = scaler.fit_transform(precos)


def criar_sequencias(dados, seq_length):
    X, y = [], []
    for i in range(len(dados) - seq_length):
        X.append(dados[i:i+seq_length])
        y.append(dados[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X, y = criar_sequencias(precos_norm, seq_length)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


model = Sequential()
model.add(LSTM(units=75, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(units=75))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))


predicoes = model.predict(X_test)
predicoes_desnorm = scaler.inverse_transform(predicoes)
y_test_desnorm = scaler.inverse_transform(y_test)


mae = mean_absolute_error(y_test_desnorm, predicoes_desnorm)
rmse = np.sqrt(mean_squared_error(y_test_desnorm, predicoes_desnorm))
r2 = r2_score(y_test_desnorm, predicoes_desnorm)


print(f"MAE (Erro Absoluto Médio): {mae:.4f}")
print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


plt.figure(figsize=(12, 6))
plt.plot(y_test_desnorm, label='Valor Real')
plt.plot(predicoes_desnorm, label='Predição LSTM')
plt.title("Comparação: Valor Real vs Predição")
plt.xlabel("Tempo")
plt.ylabel("Preço")
plt.legend()
plt.grid(True)
plt.show()


erros = y_test_desnorm - predicoes_desnorm
plt.figure(figsize=(12, 4))
plt.plot(erros, color='red', label='Erro de Predição')
plt.title("Erro de Predição (Valor Real - Previsto)")
plt.xlabel("Tempo")
plt.ylabel("Erro")
plt.legend()
plt.grid(True)
plt.show()

#explicação feita no doc apresentado junto com a entrega do trabalho