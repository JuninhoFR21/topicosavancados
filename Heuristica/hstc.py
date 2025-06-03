import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


np.random.seed(42)
precos = np.sin(np.linspace(0, 50, 500)) + np.random.normal(0, 0.5, 500)


def previsao_heuristica(precos):
    previsoes = []
    for i in range(1, len(precos)):
        variacao = precos[i] - precos[i - 1]
        previsao = precos[i] + variacao
        previsoes.append(previsao)
    return np.array(previsoes)


previsoes = previsao_heuristica(precos)


valores_reais = precos[2:]
previsoes = previsoes[1:]


mae = mean_absolute_error(valores_reais, previsoes)
rmse = np.sqrt(mean_squared_error(valores_reais, previsoes))
r2 = r2_score(valores_reais, previsoes)


media_real = np.mean(valores_reais)
acuracia_relativa = 1 - (mae / media_real)

tolerancia = 0.10
acertos = np.abs(valores_reais - previsoes) <= (tolerancia * np.abs(valores_reais))
acuracia_tolerancia = np.mean(acertos)


print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Acurácia Relativa: {acuracia_relativa:.4f}")
print(f"Acurácia com Tolerância de 10%: {acuracia_tolerancia:.4f}")


plt.figure(figsize=(12, 5))
plt.plot(valores_reais, label='Valor Real')
plt.plot(previsoes, label='Previsão Heurística')
plt.title("Baseline Heurístico - Seguir Tendência")
plt.xlabel("Tempo")
plt.ylabel("Preço")
plt.legend()
plt.grid(True)
plt.show()

erros = valores_reais - previsoes
plt.figure(figsize=(12, 4))
plt.plot(erros, color='red', label='Erro de Predição')
plt.title("Erro de Predição (Valor Real - Previsto)")
plt.xlabel("Tempo")
plt.ylabel("Erro")
plt.legend()
plt.grid(True)
plt.show()
#explicação feita no doc apresentado junto com a entrega do trabalho