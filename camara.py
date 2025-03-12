import cv2
import dlib
import numpy as np
import pyttsx3
import threading

engine = pyttsx3.init()

def decir(texto):
    threading.Thread(target=lambda: (engine.say(texto), engine.runAndWait())).start()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

EAR_THRESHOLD = 0.25

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el cuadro.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])

        left_eye = landmarks_points[36:42]
        right_eye = landmarks_points[42:48]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        if left_ear < EAR_THRESHOLD and right_ear >= EAR_THRESHOLD:
            decir("Sí")
            cv2.putText(frame, 'SI', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif right_ear < EAR_THRESHOLD and left_ear >= EAR_THRESHOLD:
            decir("No")
            cv2.putText(frame, 'NO', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
            continue

        cv2.polylines(frame, [left_eye], True, (255, 0, 0), 1)
        cv2.polylines(frame, [right_eye], True, (255, 0, 0), 1)

    cv2.imshow('Detector de Ojos', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()