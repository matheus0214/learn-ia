import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import Dense
from keras.src.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv("./iris.csv")
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

label_encoder = LabelEncoder()

# setosa     1 0 0
# virginica  0 1 0
# versicolor 0 0 1
classe = label_encoder.fit_transform(classe)
classe_dummy = to_categorical(classe)

# Divisao entre treinamento e teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = (
    train_test_split(previsores, classe_dummy, test_size=0.25)
)

classificador = Sequential()

# (qtd_entradas + saidas) / 2
classificador.add(Dense(units=4, activation="relu", input_dim=4))
classificador.add(Dense(units=4, activation="relu"))
classificador.add(Dense(units=3, activation="softmax"))
classificador.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)

classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)

resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes = previsoes > 0.5

classe_teste_2 = [np.argmax(t) for t in classe_teste]
previsoes_2 = [np.argmax(t) for t in previsoes]

cm = confusion_matrix(previsoes_2, classe_teste_2)
print(cm)
