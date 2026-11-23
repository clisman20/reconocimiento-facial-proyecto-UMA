import os
import urllib.request
import cv2

CARPETA = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(CARPETA, "face_detection_yunet_2023mar.onnx")
URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
       "face_detection_yunet/face_detection_yunet_2023mar.onnx")

if not os.path.exists(MODELO):
    print("Descargando modelo...")
    urllib.request.urlretrieve(URL, MODELO)

detector = cv2.FaceDetectorYN.create(MODELO, "", (320, 320), 0.8, 0.3, 5000)

cam = cv2.VideoCapture(0)

while True:
    ok, frame = cam.read()
    if not ok:
        break

    alto, ancho = frame.shape[:2]
    detector.setInputSize((ancho, alto))
    _, caras = detector.detect(frame)

    if caras is not None:
        for cara in caras:
            x, y, w, h = map(int, cara[:4])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Camara", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
