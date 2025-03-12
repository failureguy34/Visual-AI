import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

smoothing = 5
prev_x, prev_y = 0, 0
cam_x, cam_y = 0, 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            index_finger = landmarks[8]
            middle_finger = landmarks[12]
            thumb = landmarks[4]

            index_x = int(np.interp(index_finger.x, [0, 1], [0, screen_width]))
            index_y = int(np.interp(index_finger.y, [0, 1], [0, screen_height]))

            cur_x = prev_x + (index_x - prev_x) // smoothing
            cur_y = prev_y + (index_y - prev_y) // smoothing

            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y

            if cur_x <= 50:  # Mueve la "cámara" virtual a la izquierda
                cam_x -= 20
            elif cur_x >= screen_width - 50:  # Derecha
                cam_x += 20
            if cur_y <= 50:  # Arriba
                cam_y -= 20
            elif cur_y >= screen_height - 50:  # Abajo
                cam_y += 20

            distance_index_thumb = np.hypot(index_finger.x - thumb.x, index_finger.y - thumb.y)
            distance_middle_thumb = np.hypot(middle_finger.x - thumb.x, middle_finger.y - thumb.y)

            if distance_index_thumb < 0.05:
                pyautogui.click()
                time.sleep(0.15)
            if distance_middle_thumb < 0.05:
                pyautogui.rightClick()
                time.sleep(0.15)

    cv2.imshow('Control de Mouse con Mano', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
