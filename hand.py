import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Deshabilitar la protección contra fallos de PyAutoGUI (usa con cuidado)
pyautogui.FAILSAFE = False

# Configuración de MediaPipe para el reconocimiento de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Obtener tamaño de pantalla
screen_width, screen_height = pyautogui.size()

# Variables para controlar desplazamiento relativo
cursor_x, cursor_y = screen_width // 2, screen_height // 2
clicking = False

# Captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convertir imagen a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener coordenadas de los dedos
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Convertir a coordenadas de pantalla
            index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)

            # Calcular desplazamiento relativo
            cursor_x = np.clip(cursor_x + (index_x - w // 2) * 0.5, 0, screen_width)
            cursor_y = np.clip(cursor_y + (index_y - h // 2) * 0.5, 0, screen_height)

            pyautogui.moveTo(cursor_x, cursor_y)

            # Calcular distancia entre dedos para detectar click (índice + pulgar)
            thumb_index_dist = np.linalg.norm(np.array([thumb.x, thumb.y]) - np.array([index_finger.x, index_finger.y]))
            thumb_middle_dist = np.linalg.norm(np.array([thumb.x, thumb.y]) - np.array([middle_finger.x, middle_finger.y]))

            if thumb_index_dist < 0.05:  # Click izquierdo (índice + pulgar)
                if not clicking:
                    pyautogui.mouseDown()
                    clicking = True
            else:
                if clicking:
                    pyautogui.mouseUp()
                    clicking = False

    # Mostrar el video
    cv2.imshow('Hand Mouse', frame)

    # Salir con 'q' o 'esc'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 es el código ASCII de 'esc'
        break

cap.release()
cv2.destroyAllWindows()