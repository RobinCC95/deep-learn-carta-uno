import cv2
import numpy as np


img = cv2.imread("assets/c_2.png")
def rotarImagen(img,grados):
    rows,cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),grados,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst
# realiza una rotacion
rotacion=rotarImagen(img,45)
# guarda la imagen en la carpeta assets/rotacion_{grados}.jpg
cv2.imwrite("assets/rotacion_45.jpg",rotacion)
