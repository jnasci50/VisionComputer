import face_recognition
import os
import pickle


def train_model(image_folder):
    known_encodings = []
    known_names = []

    # Percorrer todas as pastas de pessoas
    for person_name in os.listdir(image_folder):
        person_folder = os.path.join(image_folder, person_name)

        if not os.path.isdir(person_folder):
            continue

        # Percorrer todas as imagens da pessoa
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(person_name)

    # Salvar as codificações conhecidas e os nomes em um arquivo
    with open('face_encodings.pkl', 'wb') as f:
        pickle.dump((known_encodings, known_names), f)


# Treinar o modelo com as imagens da pasta
train_model('path/to/your/image_folder')