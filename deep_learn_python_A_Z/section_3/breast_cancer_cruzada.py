import keras
import pandas as pd
from keras import Sequential
from keras.src.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv("./entradas_breast.csv")
classe = pd.read_csv("./saidas_breast.csv")


def criar_rede():
    classificador = Sequential()

    classificador.add(
        Dense(
            units=16,
            activation="relu",
            kernel_initializer="random_uniform",
            input_dim=30,
        )
    )
    classificador.add(Dropout(0.2))
    classificador.add(
        Dense(units=16, activation="relu", kernel_initializer="random_uniform")
    )
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=1, activation="sigmoid"))

    otimizador = keras.optimizers.Adam(learning_rate=0.001, decay=0.0001, clipvalue=0.5)

    classificador.compile(
        optimizer=otimizador, loss="binary_crossentropy", metrics=["binary_accuracy"]
    )

    return classificador


classificador = KerasClassifier(build_fn=criar_rede, epochs=100, batch_size=10)
resultados = cross_val_score(
    estimator=classificador, X=previsores, y=classe, cv=10, scoring="accuracy"
)
media = resultados.mean()
desvio = resultados.std()

print(media, desvio)
