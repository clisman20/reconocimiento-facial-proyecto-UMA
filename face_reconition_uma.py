import os
import json
import urllib.request
import cv2

CARPETA = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_DATOS = os.path.join(CARPETA, "personas.json")
BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/"
MODELOS = {
    "face_detection_yunet_2023mar.onnx":
        BASE + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        BASE + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

for nombre_modelo, url in MODELOS.items():
    ruta = os.path.join(CARPETA, nombre_modelo)
    if not os.path.exists(ruta):
        print("Descargando", nombre_modelo, "...")
        urllib.request.urlretrieve(url, ruta)

detector = cv2.FaceDetectorYN.create(
    os.path.join(CARPETA, "face_detection_yunet_2023mar.onnx"),
    "", (320, 320), 0.8, 0.3, 5000)
reconocedor = cv2.FaceRecognizerSF.create(
    os.path.join(CARPETA, "face_recognition_sface_2021dec.onnx"), "")

nombre = input("Nombre: ").strip()
edad = input("Edad: ").strip()
altura = input("Altura: ").strip()

cam = cv2.VideoCapture(0)
muestras = []
print("Pulsa ESPACIO para capturar (5 veces). Q para cancelar.")

while len(muestras) < 5:
    ok, frame = cam.read()
    if not ok:
        break

    alto, ancho = frame.shape[:2]
    detector.setInputSize((ancho, alto))
    _, caras = detector.detect(frame)

    vista = frame.copy()
    if caras is not None:
        for cara in caras:
            x, y, w, h = map(int, cara[:4])
            cv2.rectangle(vista, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(vista, f"Muestras {len(muestras)}/5", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Registro", vista)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord("q"):
        break
    if tecla == 32:
        if caras is None or len(caras) != 1:
            print("Debe haber exactamente un rostro.")
            continue
        alineada = reconocedor.alignCrop(frame, caras[0])
        rasgos = reconocedor.feature(alineada)
        muestras.append(rasgos.flatten().tolist())
        print("Muestra", len(muestras), "guardada")

cam.release()
cv2.destroyAllWindows()

if len(muestras) == 5:
    personas = {}
    if os.path.exists(ARCHIVO_DATOS):
        with open(ARCHIVO_DATOS, "r", encoding="utf-8") as f:
            personas = json.load(f)
    personas[nombre] = {"datos": {"edad": edad, "altura": altura},
                        "rasgos": muestras}
    with open(ARCHIVO_DATOS, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    print(nombre, "registrado.")
else:
    print("Registro cancelado.")
