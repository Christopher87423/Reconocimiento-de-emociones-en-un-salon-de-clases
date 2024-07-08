import cv2
import os
import numpy as np

# ----------- Métodos usados para el entrenamiento y lectura del modelo ----------
def load_emotion_recognizer(method):
    if method == 'EigenFaces': 
        recognizer = cv2.face.EigenFaceRecognizer_create()
    elif method == 'FisherFaces': 
        recognizer = cv2.face.FisherFaceRecognizer_create()
    elif method == 'LBPH': 
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    else:
        raise ValueError("Método de reconocimiento no válido.")
    
    model_path = f'modelo{method}.xml'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo del modelo en la ruta: {model_path}")
    
    recognizer.read(model_path)
    return recognizer

# Selecciona el método de reconocimiento
method = 'FisherFaces'
emotion_recognizer = load_emotion_recognizer(method)

# Ruta de las imágenes de datos
dataPath = 'Reconocimiento Emociones\Data'
if not os.path.exists(dataPath):
    raise FileNotFoundError(f"No se encontró el directorio de datos en la ruta: {dataPath}")
imagePaths = os.listdir(dataPath)
print('imagePaths =', imagePaths)

# Inicializa la captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise IOError("No se pudo abrir la cámara.")

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(gray):
    # Normaliza la imagen
    gray = cv2.equalizeHist(gray)
    return gray

def recognize_emotion(frame, gray, faces, recognizer, method, imagePaths):
    for (x, y, w, h) in faces:
        rostro = gray[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = recognizer.predict(rostro)
        
        # Definir los umbrales según el método
        thresholds = {'EigenFaces': 5700, 'FisherFaces': 500, 'LBPH': 60}
        threshold = thresholds.get(method, 500)

        if result[1] < threshold:
            emotion = imagePaths[result[0]].split(os.path.sep)[-1]
            cv2.putText(frame, '{}'.format(emotion), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No identificado', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = preprocess_image(gray)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    recognize_emotion(frame, gray, faces, emotion_recognizer, method, imagePaths)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
