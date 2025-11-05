import cv2
import mediapipe as mp
import joblib
import numpy as np

def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords_translated = coords - wrist
    max_dist = np.max(np.linalg.norm(coords_translated, axis=1))
    if max_dist == 0:
        return np.zeros(63).tolist()
    coords_normalized = coords_translated / max_dist
    return coords_normalized.flatten().tolist()

modelo = joblib.load('src/modelo_alfabeto.pkl')
classes = modelo.classes_

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
texto_predicao = "Nenhuma mao detectada"
CONFIDENCE_THRESHOLD = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        pontos = normalize_landmarks(hand_landmarks)
        entrada_np = np.array(pontos).reshape(1, -1)
        
        predicao = modelo.predict(entrada_np)
        probabilidade = modelo.predict_proba(entrada_np)
        confianca = np.max(probabilidade)
        
        if confianca > CONFIDENCE_THRESHOLD:
            texto_predicao = f'LETRA: {predicao[0]} ({confianca:.2f})'
        else:
            texto_predicao = 'Analisando...'
    else:
        texto_predicao = "Mostre a mao"

    cv2.putText(frame, texto_predicao, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Reconhecimento Alfabeto Libras', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()