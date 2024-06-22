import cv2
import os

# Define a pasta onde as imagens serão salvas
output_folder = 'captured_faces'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Inicializa a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Carrega o classificador de face pré-treinado do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Contador para o número de imagens capturadas
img_counter = 0

while True:
    # Captura frame por frame
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar imagem")
        break

    # Converte o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostos no frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenha retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Mostra o frame com os rostos detectados
    cv2.imshow('Face Capture', frame)

    # Aguarda a entrada do usuário
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Salva a imagem do rosto ao pressionar 'S'
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            img_name = os.path.join(output_folder, f"face_{img_counter}.png")
            cv2.imwrite(img_name, face)
            print(f"Imagem {img_name} salva!")
            img_counter += 1

    elif key == ord('x'):
        # Fecha a janela ao pressionar 'X'
        print("Encerrando captura de vídeo.")
        break

# Libera a captura e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()