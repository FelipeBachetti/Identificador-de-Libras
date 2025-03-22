import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

df = pd.read_csv("gestos.csv")

#Separando as colunas com os pontos das colunas com as labels
x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

#Codificando as labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

#80% treino e 20% teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Usando keras para criar um modelo de rede neural MLP
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)), #A ativação RELU evita problemas com o vanishing gradient
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(np.unique(y)), activation='softmax') #O softmax é útil para problemas de classificação com muitas classes
])

# compilando o model
# O sparse_categorical_crossentropy é usado quando existem múltiplas classes com rótulos inteiros sequenciais
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#treinando
model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test, y_test))

model.save("modelos_gestos.h5")

#Isso salva o modelo treinado em um arquivo pickle, que vamos usar em reconhecimento_gestos.py
import pickle
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Modelo treinado e salvo como 'modelo_gestos.h5'!")