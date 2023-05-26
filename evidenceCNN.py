import cv2
import numpy as np
from pynput import keyboard as kb
from recorte import recortar_imagen as cut

nameWindow = "Calculadora"
PATH = f"assets/carta_test.jpg"


def do_nothing(object):
    print(f"{type(object)}-{object}")
    pass


def constructor_ventana():
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min", nameWindow, 0, 255, do_nothing)
    cv2.createTrackbar("max", nameWindow, 100, 255, do_nothing)
    cv2.createTrackbar("kernel", nameWindow, 1, 100, do_nothing)
    cv2.createTrackbar("areaMin", nameWindow, 500, 10000, do_nothing)


def calcular_areas(figuras):
    areas = []
    for figuraActual in figuras:
        areas.append(cv2.contourArea(figuraActual))
    return areas


def detectar_figura(imagenOriginal):
    imagenGris = cv2.cvtColor(imagenOriginal, cv2.COLOR_BGR2GRAY)
    min = cv2.getTrackbarPos("min", nameWindow)
    max = cv2.getTrackbarPos("max", nameWindow)
    bordes = cv2.Canny(imagenGris, min, max)
    tamañoKernel = cv2.getTrackbarPos("kernel", nameWindow)
    kernel = np.ones((tamañoKernel, tamañoKernel), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = calcular_areas(figuras)
    i = 0
    areaMin = cv2.getTrackbarPos("areaMin", nameWindow)
    for figuraActual in figuras:
        if areas[i] >= areaMin:
            # Coordenadas vértices
            vertices = cv2.approxPolyDP(figuraActual, 0.05 * cv2.arcLength(figuraActual, True), True)
            if len(vertices) == 3:
                print("Es un triangulo")
                mensaje = "Triangulo" + str(areas[i])
                cv2.putText(imagenOriginal, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.drawContours(imagenOriginal, [figuraActual], 0, (0, 0, 255), 2)
            elif len(vertices) == 4:
                mensaje = "Cuadrado" + str(areas[i])
                cv2.putText(imagenOriginal, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.drawContours(imagenOriginal, [figuraActual], 0, (0, 0, 255), 2)
            elif len(vertices) == 5:
                mensaje: str = f"Pentagono{str(areas[i])}"
                cv2.putText(imagenOriginal, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.drawContours(imagenOriginal, [figuraActual], 0, (0, 0, 255), 2)
        i = i + 1
    return imagenOriginal


def do_nothing_on_press(tecla):
    pass


def soltar_p(tecla):
    if tecla == kb.KeyCode.from_char('p'):
        cv2.imwrite(PATH, frame)
        cut(cv2.imread(PATH))
        pass
    pass


video = cv2.VideoCapture(1)
constructor_ventana()

listener = kb.Listener(do_nothing_on_press, soltar_p)
listener.start()

while listener.is_alive():
    _, frame = video.read()
    #detectar_figura(frame)
    cv2.imshow("Imagen", frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
