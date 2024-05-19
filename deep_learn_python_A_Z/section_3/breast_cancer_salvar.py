import pandas as pd
from keras import Sequential
from keras.src.layers import Dense, Dropout

previsores = pd.read_csv("./entradas_breast.csv")
classe = pd.read_csv("./saidas_breast.csv")

classificador = Sequential()

classificador.add(
    Dense(
        units=8,
        activation="relu",
        kernel_initializer="normal",
        input_dim=30,
    )
)
classificador.add(Dropout(0.2))
classificador.add(Dense(units=8, activation="relu", kernel_initializer="normal"))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=1, activation="sigmoid"))

classificador.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"]
)

classificador.fit(previsores, classe, epochs=100, batch_size=10)

classificador_json = classificador.to_json()
with open("./models/classificador.json", "w") as json_file:
    json_file.write(classificador_json)

classificador.save_weights("./models/classificador.weights.h5")
