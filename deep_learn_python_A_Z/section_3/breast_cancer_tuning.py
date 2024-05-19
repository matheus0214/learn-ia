import pandas as pd
from keras import Sequential
from keras.src.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv("./entradas_breast.csv")
classe = pd.read_csv("./saidas_breast.csv")


def criar_rede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()

    classificador.add(
        Dense(
            units=neurons,
            activation=activation,
            kernel_initializer=kernel_initializer,
            input_dim=30,
        )
    )
    classificador.add(Dropout(0.2))
    classificador.add(
        Dense(
            units=neurons, activation=activation, kernel_initializer=kernel_initializer
        )
    )
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=1, activation="sigmoid"))

    classificador.compile(optimizer=optimizer, loss=loss, metrics=["binary_accuracy"])

    return classificador


classificador = KerasClassifier(build_fn=criar_rede)
parametros = {
    "model__optimizer": ["adam", "sgd"],
    "epochs": [30, 50],
    "batch_size": [10, 30],
    "model__kernel_initializer": ["random_uniform", "normal"],
    "model__activation": ["relu", "tanh"],
    "model__neurons": [16, 8],
    "model__loss": ["binary_crossentropy", "hinge"],
}

grid_search = GridSearchCV(
    estimator=classificador, param_grid=parametros, scoring="accuracy", cv=5
)
grid_search = grid_search.fit(previsores, classe)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

print(melhores_parametros, melhor_precisao)
