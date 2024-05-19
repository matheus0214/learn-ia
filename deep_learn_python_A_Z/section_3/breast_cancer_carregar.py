import numpy as np
import pandas as pd
from keras.src.models.model import model_from_json

arquivo = open("./models/classificador.json", "r")
estrutura = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura)
classificador.load_weights("./models/classificador.weights.h5")


def classificar_um_registro(data):
    novo = np.array(
        [
            [
                15.80,
                8.34,
                118,
                900,
                0.10,
                0.26,
                0.08,
                0.134,
                0.178,
                0.20,
                0.05,
                1098,
                0.87,
                4500,
                145.2,
                0.005,
                0.04,
                0.05,
                0.015,
                0.03,
                0.007,
                23.15,
                16.64,
                178.5,
                2018,
                0.14,
                0.185,
                0.84,
                158,
                0.363,
            ]
        ]
    )
    previsao = classificador.predict(np.array([data]))

    return float(previsao[0][0])


def classificar_muitos_registros():
    previsores = pd.read_csv("./entradas_breast.csv")
    classe = pd.read_csv("./saidas_breast.csv")

    classificador.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    classificador.fit(previsores, classe, epochs=100, batch_size=10)

    resultado = classificador.evaluate(previsores, classe)
    print(resultado)
