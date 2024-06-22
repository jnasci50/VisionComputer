import cv2
import face_recognition
import pickle
import numpy as np

# Carregar o classificador Haarcascade para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carregar o modelo treinado
with open('face_encodings.pkl', 'rb') as f:
    known_encodings, known_names = pickle.load(f)


def recognize_faces(frame, known_encodings, known_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # Usar o rosto mais próximo se houver uma correspondência
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]

        # Desenhar um retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Desenhar o nome abaixo do rosto
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    return frame


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

    # Reconhecer rostos no frame
    frame = recognize_faces(frame, known_encodings, known_names)

    # Exibir o frame
    cv2.imshow('Webcam - Reconhecimento Facial', frame)

    # Pressionar 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar todas as janelas OpenCV
cap.release()
cv2.destroyAllWindows()