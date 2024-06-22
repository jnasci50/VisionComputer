import cv2
import numpy as np


def preprocess_image(img):
    # Redimensionar a imagem para 224x224 pixels
    img = cv2.resize(img, (224, 224))

    # Converter a imagem de BGR para RGB (OpenCV carrega a imagem em BGR por padrão)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalizar a imagem de acordo com a ResNet-50
    img = img.astype('float32')
    mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    img -= mean

    # Transpor a imagem de HWC para CHW
    img = np.transpose(img, (2, 0, 1))

    # Adicionar uma dimensão para criar um batch de tamanho 1
    img_batch = np.expand_dims(img, axis=0)

    return img_batch


# Carregar o modelo ResNet-50 ONNX com OpenCV
model_path = 'Resnet_Data/resnet50-v2-7.onnx'
net = cv2.dnn.readNetFromONNX(model_path)

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

    # Processar a imagem
    processed_frame = preprocess_image(frame)
    print(processed_frame)
    # Fazer a previsão usando ResNet-50
    net.setInput(processed_frame)
    preds = net.forward()

    # Decodificar as previsões
    # Para simplicidade, carregamos as etiquetas do ImageNet de um arquivo
    with open("Resnet_Data/imagenet-labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # Obter as top 3 previsões
    top_indices = np.argsort(preds[0])[::-1][:3]
    top_labels = [labels[i] for i in top_indices]
    top_scores = [preds[0][i] for i in top_indices]

    # Exibir as previsões na imagem
    for i, (label, score) in enumerate(zip(top_labels, top_scores)):
        text = f"{label}: {score:.2f}"
        cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f'imagem para resnet antes da webcam:{text}')

    # Exibir o frame
    cv2.imshow('Webcam', frame)

    # Pressionar 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Liberar a captura de vídeo e fechar todas as janelas OpenCV
cap.release()
cv2.destroyAllWindows()