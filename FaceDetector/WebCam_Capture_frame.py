import cv2
import os
from PIL import Image

# Define a pasta onde as imagens serão salvas
output_folder = 'captured_faces'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Carregar o classificador Haarcascade para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Contador para o número de imagens capturadas
img_counter = 0

# Inicializar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

while True:
    # Capturar frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar a imagem.")
        break

    # Detectar rostos no frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenhar retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Exibir o frame
    cv2.imshow('Webcam - Detecção Facial', frame)

    # Aguarda a entrada do usuário
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Salva a imagem do rosto ao pressionar 'S'
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]

            # Salvar a imagem em escala de cinza 8 bits
            img_name_gray = os.path.join(output_folder, f"face_{img_counter}_gray.png")
            cv2.imwrite(img_name_gray, cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
            print(f"Imagem {img_name_gray} salva!")

            # Salvar a imagem em RGB
            img_name_rgb = os.path.join(output_folder, f"face_{img_counter}_rgb.png")
            cv2.imwrite(img_name_rgb, cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            print(f"Imagem {img_name_rgb} salva!")

            img_counter += 1

    elif key == ord('x'):
        # Fecha a janela ao pressionar 'X'
        print("Encerrando captura de vídeo.")
        break

# Liberar a captura de vídeo e fechar todas as janelas OpenCV
cap.release()
cv2.destroyAllWindows()