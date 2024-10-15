# Importaciones necesarias
import numpy as np
import cv2
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# 1. Generadores de datos

# Aumento de datos solo para el conjunto de entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalizar imágenes
    rotation_range=20,        # Rotaciones aleatorias de hasta 20 grados
    zoom_range=0.2,           # Zoom aleatorio en las imágenes
    width_shift_range=0.1,    # Desplazamiento horizontal aleatorio
    height_shift_range=0.1,   # Desplazamiento vertical aleatorio
    horizontal_flip=True,     # Inversiones horizontales
    brightness_range=[0.9, 1.1],  # Variaciones de brillo
    fill_mode='nearest'       # Completa con los valores más cercanos
)

# Normalización (sin aumento) para el conjunto de validación/pruebas
test_datagen = ImageDataGenerator(rescale=1./255)

# Carga las imágenes del conjunto de entrenamiento
train_generator = train_datagen.flow_from_directory(
    'productos/productos_train/',    # Ruta a las imágenes de entrenamiento
    target_size=(150, 150),          # Redimensionar las imágenes a 150x150
    batch_size=32,                   # Tamaño del lote
    class_mode='categorical'         # Clasificación multiclase
)

# Carga las imágenes del conjunto de prueba
test_generator = test_datagen.flow_from_directory(
    'productos/productos_test/',     # Ruta a las imágenes de prueba
    target_size=(150, 150),          # Redimensionar las imágenes a 150x150
    batch_size=32,                   # Tamaño del lote
    class_mode='categorical'         # Clasificación multiclase
)

# 2. Construcción del modelo CNN

def construir_modelo():
    model = Sequential([
        # Capa de entrada con tamaño de imagen 150x150x3 (3 canales RGB)
        Input(shape=(150, 150, 3)),
        # Primera capa convolucional con 32 filtros de 3x3
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),  # Submuestreo de 2x2
        # Segunda capa convolucional con 64 filtros de 3x3
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        # Tercera capa convolucional con 128 filtros de 3x3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        # Cuarta capa convolucional con 256 filtros de 3x3
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        # Aplanar las características extraídas por las capas convolucionales
        Flatten(),
        # Capa densa totalmente conectada con 256 unidades y activación ReLU
        Dense(256, activation='relu'),
        # Añadir Dropout para evitar el sobreajuste
        Dropout(0.5),
        # Otra capa densa con 128 unidades
        Dense(128, activation='relu'),
        # Capa de salida con 4 unidades (una por cada clase) y activación softmax
        Dense(4, activation='softmax')  # 4 categorías: bananas, frutillas, manzanas, sandías
    ])

    # Compilación del modelo con optimizador Adam y función de pérdida categorical_crossentropy
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss='categorical_crossentropy',
        metrics=['accuracy']  # Métrica de precisión
    )
    return model

# 3. Entrenamiento del modelo

# Callback para detener el entrenamiento si no hay mejora en la pérdida de validación
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Construir el modelo
model = construir_modelo()

# Entrenar el modelo
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Número de pasos por época
    epochs=50,                             # Número de épocas
    validation_data=test_generator,        # Datos de validación
    validation_steps=len(test_generator),  # Número de pasos en la validación
    callbacks=[early_stopping]             # Callback para detener temprano
)

# Guardar el modelo entrenado
model.save('model_fruta.keras')

# 4. Evaluación del modelo

# Evaluar el modelo en los datos de prueba
test_loss, test_acc = model.evaluate(test_generator)
print(f"Precisión en datos de prueba: {test_acc:.2f}")

# 5. Reporte de clasificación y matriz de confusión

# Generar predicciones en los datos de prueba
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)  # Obtener la clase predicha
y_true = test_generator.classes             # Obtener las clases reales

# Imprimir matriz de confusión
print(confusion_matrix(y_true, y_pred_classes))

# Imprimir reporte de clasificación (precisión, recall, F1)
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

# 6. Predicción en tiempo real con OpenCV

# Predicción en tiempo real con OpenCV
def predecir_en_tiempo_real(model, train_generator):
    """
    Función que captura imágenes de la cámara en tiempo real y realiza predicciones continuamente.
    """
    camera = cv2.VideoCapture(0)  # Iniciar la cámara (usar '0' para la cámara por defecto)
    
    if not camera.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return
    
    while True:
        ret, frame = camera.read()  # Leer un cuadro de la cámara
        
        if not ret:
            print("Error al capturar imagen.")
            break
        
        # Redimensionar la imagen a (150, 150) para que coincida con el tamaño de entrada del modelo
        imagen_resized = cv2.resize(frame, (150, 150))
        
        # Convertir la imagen a array y expandir dimensiones para que sea compatible con el modelo
        img_array = img_to_array(imagen_resized) / 255.0  # Normalización de la imagen
        img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para que sea (1, 150, 150, 3)
        
        # Realizar predicción
        prediccion = model.predict(img_array)
        
        # Obtener la clase con la probabilidad más alta
        indice_clase = np.argmax(prediccion)
        clase_predicha = list(train_generator.class_indices.keys())[indice_clase]
        
        # Agregar un marco verde alrededor de la imagen
        alto, ancho, _ = frame.shape
        inicio_x, inicio_y = 10, 10  # Esquina superior izquierda
        fin_x, fin_y = ancho - 10, alto - 10  # Esquina inferior derecha
        cv2.rectangle(frame, (inicio_x, inicio_y), (fin_x, fin_y), (0, 255, 0), 2)

        # Mostrar el cuadro de video con la clase predicha en tiempo real
        cv2.putText(frame, f"Clase: {clase_predicha}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Predicción en tiempo real', frame)
        
        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar la cámara y cerrar las ventanas
    camera.release()
    cv2.destroyAllWindows()

# Llamar a la función para iniciar la predicción en tiempo real
#predecir_en_tiempo_real()
