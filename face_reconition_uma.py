import os
import json
import unicodedata
import urllib.request
import cv2
import numpy as np

CARPETA = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_DATOS = os.path.join(CARPETA, "personas.json")
UMBRAL = 0.363
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


def sin_acentos(texto):
    texto = unicodedata.normalize("NFD", str(texto))
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def cargar_personas():
    if not os.path.exists(ARCHIVO_DATOS):
        return {}
    with open(ARCHIVO_DATOS, "r", encoding="utf-8") as f:
        return json.load(f)


def guardar_personas(personas):
    with open(ARCHIVO_DATOS, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)


def detectar(frame):
    alto, ancho = frame.shape[:2]
    detector.setInputSize((ancho, alto))
    _, caras = detector.detect(frame)
    return [] if caras is None else caras


def sacar_rasgos(frame, cara):
    alineada = reconocedor.alignCrop(frame, cara)
    return reconocedor.feature(alineada)


def registrar():
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

        caras = detectar(frame)
        vista = frame.copy()
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
            if len(caras) != 1:
                print("Debe haber exactamente un rostro.")
                continue
            rasgos = sacar_rasgos(frame, caras[0])
            muestras.append(rasgos.flatten().tolist())
            print("Muestra", len(muestras), "guardada")

    cam.release()
    cv2.destroyAllWindows()

    if len(muestras) == 5:
        personas = cargar_personas()
        personas[nombre] = {"datos": {"edad": edad, "altura": altura},
                            "rasgos": muestras}
        guardar_personas(personas)
        print(nombre, "registrado.")
    else:
        print("Registro cancelado.")


def reconocer():
    personas = cargar_personas()
    if not personas:
        print("No hay personas registradas. Usa la opcion 1 primero.")
        return

    conocidos = {
        nombre: [np.array(r, dtype=np.float32).reshape(1, -1)
                 for r in info["rasgos"]]
        for nombre, info in personas.items()
    }

    cam = cv2.VideoCapture(0)
    presentes_antes = set()
    print("Reconociendo... pulsa Q en la ventana para salir.")

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        presentes_ahora = set()
        for cara in detectar(frame):
            x, y, w, h = map(int, cara[:4])
            rasgos = sacar_rasgos(frame, cara)

            mejor_nombre = None
            mejor_puntaje = -1.0
            for nombre, lista in conocidos.items():
                for guardado in lista:
                    puntaje = reconocedor.match(
                        rasgos, guardado, cv2.FaceRecognizerSF_FR_COSINE)
                    if puntaje > mejor_puntaje:
                        mejor_nombre = nombre
                        mejor_puntaje = puntaje

            if mejor_puntaje >= UMBRAL:
                datos = personas[mejor_nombre]["datos"]
                presentes_ahora.add(mejor_nombre)
                color = (0, 200, 0)
                lineas = [mejor_nombre] + [f"{k}: {v}" for k, v in datos.items()]
                if mejor_nombre not in presentes_antes:
                    print("Identificando... a", mejor_nombre)
                    for campo, valor in datos.items():
                        print(f"  {campo}: {valor}")
            else:
                color = (0, 0, 255)
                lineas = ["Desconocido"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            for i, linea in enumerate(lineas):
                cv2.putText(frame, sin_acentos(linea), (x, y + h + 22 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        presentes_antes = presentes_ahora
        cv2.imshow("Reconocimiento facial", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


while True:
    print("\n1. Registrar persona")
    print("2. Reconocer")
    print("0. Salir")
    opcion = input("Elige una opcion: ").strip()
    if opcion == "1":
        registrar()
    elif opcion == "2":
        reconocer()
    elif opcion == "0":
        break
    else:
        print("Opcion no valida.")