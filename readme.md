# Reconocimiento facial con Python y OpenCV

Proyecto de primer semestre de Ingeniería Mecatrónica.
Autor: Clisman Francisco Maquirino
y otros

## Versiones

- **v1.0**: abre la cámara y muestra el video en vivo.
- **v2.0**: detecta rostros y dibuja un cuadro verde.
- **v3.0**: registra personas: guarda su nombre, datos y huella facial en un JSON.
- **v4.0**: reconoce personas y muestra sus datos en pantalla.

## Cómo ejecutarlo

    pip install opencv-python numpy
    python face_reconition_uma.py

La primera vez descarga los modelos (necesita internet).

## Créditos

Modelos YuNet (detección) y SFace (reconocimiento) de OpenCV Zoo:
https://github.com/opencv/opencv_zoo