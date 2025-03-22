import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

df = pd.read_csv("gestos.csv")

print("Dataset shape:", df.shape)

#Separando as colunas com os pontos das colunas com as labels
feat = df.iloc[:, 1:].values
labels = df.iloc[:, 0].values

#Codificando as labels
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

#80% treino e 20% teste
x_train, x_test, y_train, y_test = train_test_split(feat, labels_encoded, test_size=0.2, random_state=42)

#Usando keras para criar um modelo de rede neural MLP
model = keras.Sequential([
    keras.layers.Dense(120, activation='relu', input_shape=(x_train.shape[1],)),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(len(np.unique(labels_encoded)), activation='softmax')
])

# compilando o model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#treinando
model.fit(x_train, y_train, epochs=55, batch_size=20, validation_data=(x_test, y_test))

model.save("modelos_gestos.h5")

import pickle
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Modelo treinado e salvo")