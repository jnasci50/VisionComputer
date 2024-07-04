import face_recognition
import os
import pickle
from PIL import Image
import numpy as np

def load_image_file(file):
    """
    Carrega uma imagem e a converte para RGB se necessário.
    """
    image = Image.open(file)
    image = image.convert('RGB')  # Forçar a conversão para RGB
    return np.array(image)

def train_model(image_folder):
    known_encodings = []
    known_names = []

    # Verificar se a pasta de imagens existe
    if not os.path.exists(image_folder):
        print(f"A pasta {image_folder} não existe.")
        return

    # Percorrer todas as pastas de pessoas
    for person_name in os.listdir(image_folder):
        person_folder = os.path.join(image_folder, person_name)

        if not os.path.isdir(person_folder):
            continue

        # Percorrer todas as imagens da pessoa
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            print(f"Processando {image_path}")

            try:
                # Carregar a imagem e converter para RGB
                image_array = load_image_file(image_path)

                encodings = face_recognition.face_encodings(image_array)

                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                else:
                    print(f"Sem codificação encontrada para {image_path}")
            except Exception as e:
                print(f"Erro ao processar {image_path}: {e}")

    # Verificar se foram encontradas codificações
    if len(known_encodings) == 0:
        print("Nenhuma codificação encontrada.")

    # Salvar as codificações conhecidas e os nomes em um arquivo
    with open('face_encodings.pkl', 'wb') as f:
        pickle.dump((known_encodings, known_names), f)
    print("Modelo treinado e salvo como face_encodings.pkl")

# Treinar o modelo com as imagens da pasta
train_model('captured_faces')