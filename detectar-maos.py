import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time

# Isso vai inicializar o detector de maos do MediaPipe
# O mídia pipe é uma framework capaz de captar elementos em imagens e vídeos, muito útil para a visão computacional
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

#Esse modelo vai detectar um maximo de duas maos em tempo real
detector_maos = mp_maos.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Nome do gesto a ser capturado (Para o treinamento)
gesto_nome = input("Nome do gesto: ")
dados = []

#Vai capturar o vídeo pela camera padrão
cap = cv2.VideoCapture(0)
tempo_inicio = time.time()
tempo_limite = 30

print(f"Capturando o gesto '{gesto_nome}' por {tempo_limite} segundos...")

while cap.isOpened():
    # Vamos trabalhar com cada frame
    ret, frame = cap.read()
    if not ret:
        break

    # O media pipe precisa da imagem em RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Processamento
    resultado = detector_maos.process(frame_rgb)

    if resultado.multi_hand_landmarks:
        for landmarks in resultado.multi_hand_landmarks:
            #Pega as coordenadas do gesto e armazena
            pontos = []
            for lm in landmarks.landmark:
                pontos.append(lm.x)
                pontos.append(lm.y)
                pontos.append(lm.z)     
            dados.append([gesto_nome] + pontos)

            #Desenha as landmarks (pontos importantes)
            mp_desenho.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)

    cv2.imshow("Deteccao de Maos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#Salva os dados de treinamento
colunas = ["gesto"] + [f"ponto_{i}" for i in range(len(dados[0]) - 1)]
df = pd.DataFrame(dados, columns=colunas)
df.to_csv("gestos.csv", mode='a', header=not pd.io.common.file_exists("gestos.csv"), index=False)

print(f"Dados do gesto '{gesto_nome}' salvos em 'gestos.csv'!")