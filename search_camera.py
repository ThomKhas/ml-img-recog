import cv2

def probar_camaras(max_index=5):
    """
    Intenta abrir cámaras en los índices desde 0 hasta max_index y muestra cuál funciona.
    """
    for i in range(max_index + 1):
        print(f"Probando cámara en el índice {i}...")
        camera = cv2.VideoCapture(i)
        if camera.isOpened():
            print(f"¡Cámara encontrada en el índice {i}! Mostrando vista previa...")

            while True:
                ret, frame = camera.read()
                if not ret:
                    print("Error al capturar el cuadro.")
                    break

                # Agregar un marco verde alrededor de la imagen
                alto, ancho, _ = frame.shape
                # Definir las coordenadas del rectángulo
                inicio_x, inicio_y = 100, 100  # Esquina superior izquierda
                fin_x, fin_y = ancho - 100, alto - 100  # Esquina inferior derecha
                # Dibujar el rectángulo
                cv2.rectangle(frame, (inicio_x, inicio_y), (fin_x, fin_y), (0, 255, 0), 2)

                # Muestra el fotograma en una ventana
                cv2.imshow(f'Cámara {i}', frame)

                # Espera 1 ms y verifica si se presiona la tecla 'q' para salir
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            camera.release()
            cv2.destroyAllWindows()
            return i
        else:
            print(f"No se pudo abrir la cámara en el índice {i}.")
    
    print("No se encontró ninguna cámara disponible.")
    return None

# Ejecutar la prueba
indice_camara = probar_camaras(max_index=5)

if indice_camara is not None:
    print(f"El índice correcto de la cámara es: {indice_camara}")
else:
    print("No se pudo encontrar una cámara disponible.")
