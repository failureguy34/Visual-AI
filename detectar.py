import cv2

print("Buscando cámaras disponibles...")

# Probar índices del 0 al 9
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Cámara encontrada en el índice {i}")
        cap.release()

print("Búsqueda completada.")
