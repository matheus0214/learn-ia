import pandas as pd
from keras import Input, Model
from keras.layers import Dense
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

base = pd.read_csv("games.csv")
base.drop("Other_Sales", axis=1, inplace=True)
base.drop("Global_Sales", axis=1, inplace=True)
base.drop("Developer", axis=1, inplace=True)

base.dropna(axis=0, inplace=True)
base = base.loc[base["NA_Sales"] > 1]
base = base.loc[base["EU_Sales"] > 1]

nome_jogos = base["Name"]

base.drop("Name", axis=1, inplace=True)
# print(base.shape)

previsores = base.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11]].values
venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values

labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])

onehotencoder = ColumnTransformer(
    transformers=[("OneHot", OneHotEncoder(), [0, 2, 3, 8])],
    remainder="passthrough",
)

previsores = onehotencoder.fit_transform(previsores).toarray()

camada_entrada = Input(shape=(61,))

camada_oculta_1 = Dense(32, activation="sigmoid")(camada_entrada)
camada_oculta_2 = Dense(32, activation="sigmoid")(camada_oculta_1)

camada_saida_1 = Dense(units=1, activation="linear")(camada_oculta_2)
camada_saida_2 = Dense(units=1, activation="linear")(camada_oculta_2)
camada_saida_3 = Dense(units=1, activation="linear")(camada_oculta_2)

regressor = Model(
    inputs=camada_entrada, outputs=[camada_saida_1, camada_saida_2, camada_saida_3]
)

regressor.compile(loss="mse", optimizer="adam")
regressor.fit(previsores, [venda_na, venda_eu, venda_jp], epochs=5000, batch_size=100)

previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)

print(previsao_na, previsao_eu, previsao_jp)
