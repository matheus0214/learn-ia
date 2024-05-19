import keras
import pandas as pd
from keras.api.layers import Dense
from keras.api.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

previsores = pd.read_csv("./entradas_breast.csv")
classe = pd.read_csv("./saidas_breast.csv")

# Divisao da base de dados de teste e treinamento
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = (
    train_test_split(previsores, classe, test_size=0.25)
)

classificador = Sequential()

# (entradas + saidas) / 2
# (30 + 1)/2

classificador.add(
    Dense(
        units=16, activation="relu", kernel_initializer="random_uniform", input_dim=30
    )
)
classificador.add(
    Dense(units=16, activation="relu", kernel_initializer="random_uniform")
)

classificador.add(Dense(units=1, activation="sigmoid"))

otimizador = keras.optimizers.Adam(learning_rate=0.001, decay=0.0001, clipvalue=0.5)

classificador.compile(
    optimizer=otimizador, loss="binary_crossentropy", metrics=["binary_accuracy"]
)

classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)

previsoes = classificador.predict(previsores_teste)
previsoes = previsoes > 0.5

precisao = accuracy_score(classe_teste, previsoes)
cm = confusion_matrix(classe_teste, previsoes)
resultado = classificador.evaluate(previsores_teste, classe_teste)

print(f"Precisao: {precisao}")
print(f"Resultado: {resultado}")
print(cm)
