from keras import Sequential
from keras.api.datasets import mnist
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.utils import to_categorical

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

# plt.imshow(X_treinamento[1], cmap=plt.cm.gray)
# plt.title(f"Classe {X_treinamento[0]}")
# plt.show()

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype("float32") / 255

previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_teste = previsores_teste.astype("float32") / 255

classe_treinamento = to_categorical(y_treinamento, 10)
classe_teste = to_categorical(y_teste)

classificador = Sequential()

classificador.add(
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1))
)
classificador.add(MaxPooling2D(pool_size=(2, 2)))
classificador.add(Flatten())

classificador.add(Dense(units=128, activation="relu"))
classificador.add(Dense(units=10, activation="softmax"))

classificador.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

classificador.fit(
    previsores_treinamento,
    classe_treinamento,
    batch_size=128,
    epochs=5,
    validation_data=(previsores_teste, classe_teste),
)

resultado = classificador.evaluate(previsores_teste, classe_teste)
