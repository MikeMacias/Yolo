import cv2
from models import Yolov4

model = Yolov4(weight_path='yolov4.weights',
               class_name_path='class_names/coco_classes.txt')

# Lee el archivo de video MP4
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('img/cat.mp4')

while True:
    # Lee un cuadro del video
    ret, frame = cap.read()
    if not ret:
        break

    # Detecta objetos en el cuadro utilizando YOLOv4
    pred = model.predict(frame)

    # Muestra el cuadro con los objetos detectados
    cv2.imshow("nueva", model.output_img)

    # Espera por un tecla para salir del bucle
    if cv2.waitKey(1) == ord('q'):
        break

# Limpia la memoria y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()