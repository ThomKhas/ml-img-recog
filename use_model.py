import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Cargar el modelo entrenado
model = load_model('model_fruta.keras')

# Definir las clases (asegúrate de que estén en el mismo orden que durante el entrenamiento)
clases = ['bananas', 'frutillas', 'manzanas', 'sandias']  # Ajusta esto según tus clases reales

# Umbral de confianza para la predicción
UMBRAL_CONFIANZA = 0.7  # Ajusta este valor según sea necesario

def predecir_imagen(model, imagen, clases):
    """
    Realiza la predicción en una imagen dada.
    """
    imagen_resized = cv2.resize(imagen, (150, 150))
    img_array = img_to_array(imagen_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediccion = model.predict(img_array)
    indice_clase = np.argmax(prediccion)
    confianza = np.max(prediccion)
    
    if confianza > UMBRAL_CONFIANZA:
        return clases[indice_clase], confianza
    else:
        return "No hay coincidencias", confianza

def predecir_con_espacio(model, clases):
    """
    Función que captura imágenes de la cámara y realiza predicciones cuando se presiona la tecla espacio.
    """
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return
    
    clase_predicha = "Presiona ESPACIO para predecir"
    confianza = 0.0
    
    while True:
        ret, frame = camera.read()
        
        if not ret:
            print("Error al capturar imagen.")
            break
        
        # Agregar un marco verde alrededor de la imagen
        alto, ancho, _ = frame.shape
        cv2.rectangle(frame, (10, 10), (ancho - 10, alto - 10), (0, 255, 0), 2)

        # Mostrar instrucciones y predicción actual
        cv2.putText(frame, "Presiona ESPACIO para predecir", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Clase: {clase_predicha}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confianza: {confianza:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Predicción con ESPACIO', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Tecla espacio
            clase_predicha, confianza = predecir_imagen(model, frame, clases)
    
    camera.release()
    cv2.destroyAllWindows()

# Llamar a la función para iniciar la predicción
predecir_con_espacio(model, clases)