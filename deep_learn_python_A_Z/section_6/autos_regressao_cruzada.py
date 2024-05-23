import pandas as pd
from keras import Sequential
from keras.src.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

base = pd.read_csv("autos.csv", encoding="ISO-8859-1")
base = base.drop("dateCrawled", axis=1)
base = base.drop("dateCreated", axis=1)
base = base.drop("nrOfPictures", axis=1)
base = base.drop("postalCode", axis=1)
base = base.drop("lastSeen", axis=1)
base = base.drop("name", axis=1)
base = base.drop("seller", axis=1)
base = base.drop("offerType", axis=1)

base = base[base.price > 10]
base = base.loc[base.price < 350000]

valores = {
    "vehicleType": "limousine",
    "gearbox": "manuell",
    "model": "golf",
    "fuelType": "benzin",
    "notRepairedDamage": "nein",
}

base = base.fillna(value=valores)

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

labelencoder_previsores = LabelEncoder()

previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])

one_hot_encoder = ColumnTransformer(
    transformers=[("OneHot", OneHotEncoder(), [0, 1, 3, 5, 8, 9, 10])],
    remainder="passthrough",
)
previsores = one_hot_encoder.fit_transform(previsores).toarray()


def criar_rede():
    regressor = Sequential()

    regressor.add(Dense(units=158, activation="relu", input_dim=316))
    regressor.add(Dense(units=158, activation="relu"))
    regressor.add(Dense(units=1, activation="linear"))
    regressor.compile(
        optimizer="adam", loss="mean_absolute_error", metrics=["mean_absolute_error"]
    )

    return regressor


regressor = KerasRegressor(build_fn=criar_rede, epochs=100, batch_size=300)
resultados = cross_val_score(
    estimator=regressor,
    X=previsores,
    y=preco_real,
    cv=10,
    scoring="neg_root_mean_squared_log_error",
)

media = resultados.mean()
desvio = resultados.std()
print(f"Media: {media}\nDesvio: {desvio}")
