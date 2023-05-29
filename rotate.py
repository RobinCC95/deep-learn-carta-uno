import cv2
import numpy as np

ruta = "dataset/train"
ruta_origin = "assets/image"
def rotarImagen(img, grados, tamanio=(128, 128)):
    """
    funcion para rotar una imagen
    :param img: imagen a rotar
    :param grados: grados a rotar la imagen
    :param tamanio: tamaño de la imagen
    :return:  imagen rotada
    """
    rows, cols = img.shape[:2]
    # Calcula el tamaño del cuadro de límite después de la rotación
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), grados, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_cols = int((rows * sin) + (cols * cos))
    new_rows = int((rows * cos) + (cols * sin))

    # Ajusta la matriz de rotación para evitar recorte
    M[0, 2] += (new_cols / 2) - (cols / 2)
    M[1, 2] += (new_rows / 2) - (rows / 2)

    # Aplica la rotación a la imagen sin recortar
    dst = cv2.warpAffine(img, M, (new_cols, new_rows))
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    #redimentciona la imagen
    img_redim = cv2.resize(gray, tamanio)
    return img_redim

def save_dataset(ruta, rut_img_ini, limite= 10, pasos_grados= 5):
    """
    funcion para guardar el dataset de imagenes rotadas en una carpeta

    :param ruta: ruta donde se guardara el dataset
    :param rut_img_ini:  ruta de las imagenes originales
    :param limite:  limite de imagenes a rotar
    :param pasos_grados:  pasos de grados a rotar
    :return:  None
    """
    for categoria in range (limite):
        ruta_origin_img = rut_img_ini+"/"+ str(categoria)+ ".png"
        print(ruta_origin_img)
        for grados in range (0,360,pasos_grados):
            ruta_save = ruta+"/"+str(categoria)+"/"+str(categoria)+"_"+str(grados)+".jpg"
            print(ruta_save)
            img = cv2.imread(ruta_origin_img)
            cv2.imwrite(ruta_save, rotarImagen(img, grados))


save_dataset(ruta, ruta_origin, pasos_grados= 1)


# Ejemplo de uso
# img = cv2.imread("assets/c_1.png")
# rotacion = rotarImagen(img, 35)
# cv2.imwrite("assets/rotacion_70.jpg", rotacion)
