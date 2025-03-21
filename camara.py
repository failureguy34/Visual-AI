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
prev_x, prev_y = 0, 0
cursor_x, cursor_y = screen_width // 2, screen_height // 2

# Variables para el arrastre (drag)
dragging = False

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
            thumb_finger = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convertir a coordenadas de pantalla
            index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)
            thumb_x, thumb_y = int(thumb_finger.x * w), int(thumb_finger.y * h)

            # Calcular desplazamiento relativo
            dx = index_x - prev_x
            dy = index_y - prev_y

            # Actualizar posición del cursor (permitiendo arrastre más allá de los límites de la cámara)
            cursor_x = np.clip(cursor_x + dx * 2, 0, screen_width)
            cursor_y = np.clip(cursor_y + dy * 2, 0, screen_height)

            pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)

            # Verificar distancia entre pulgar e índice (para clic izquierdo o arrastre)
            distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)
            if distance < 40:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # Actualizar las coordenadas previas
            prev_x, prev_y = index_x, index_y

    # Mostrar el video
    cv2.imshow('Hand Mouse', frame)

    # Salir con 'q' o 'esc'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 es el código ASCII de 'esc'
        break

cap.release()
cv2.destroyAllWindows()