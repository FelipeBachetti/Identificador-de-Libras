import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

#carregando o modelo
model = tf.keras.models.load_model("modelos_gestos.h5")
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
detector_maos = mp_maos.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = detector_maos.process(frame_rgb)

    gesto_predito = "Nenhum"

    if resultado.multi_hand_landmarks:
        for landmarks in resultado.multi_hand_landmarks:
            if len(landmarks.landmark) == 21:
                pontos = []
                for lm in landmarks.landmark:
                    pontos.append(lm.x)
                    pontos.append(lm.y)
                    pontos.append(lm.z)

                #previsão do gesto
                pontos = np.array(pontos).reshape(1, -1)

                if pontos.shape[1] == 63:
                    previsao = model.predict(pontos)
                    indice_gesto = np.argmax(previsao) #maior prob
                    gesto_predito = encoder.inverse_transform([indice_gesto])[0] #nome do gesto
                else:
                    gesto_predito = "Erro na entrada"

                #desenha a mao e escreve o nome
                mp_desenho.draw_landmarks(frame, landmarks, mp_maos.HAND_CONNECTIONS)
                cv2.putText(frame, gesto_predito, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Reconhecimento de Gestos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()