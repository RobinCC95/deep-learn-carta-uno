import cv2
import numpy as np
import random
ruta = "dataset/train"
ruta_origin = "assets/data"
ruta_origin_test = "assets/image_test"


def scale_tamani_gris(img, tamanio=(128, 128)):
    """
    funcion para redimencionar una imagen a escala de grises
    :param img: imagen a redimencionar
    :param tamanio: tamaño de la imagen
    :return: imagen redimencionada
    """
    image = cv2.resize(img, tamanio)
    img_out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img_out

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
    img_out = scale_tamani_gris(dst, tamanio)
    return img_out

def translate_image(image, shift_x, shift_y):
    # img_out = scale_tamani_gris(image)
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated_image = cv2.warpAffine(image, M, (cols, rows))
    return translated_image

def save_dataset_test(ruta, rut_img_ini, limite= 10):
    """
    funcion para guardar el dataset de imagenes rotadas en una carpeta

    :param ruta: ruta donde se guardara el dataset
    :param rut_img_ini:  ruta de las imagenes originales
    :param limite:  limite de imagenes a rotar
    :param pasos_grados:  pasos de grados a rotar
    :return:  None
    """
    for categoria in range (limite):
        ruta_origin_img = rut_img_ini+"/"+ str(categoria)+"_"+str(random.randint(0, 3)) +".jpg"
        print(ruta_origin_img)
        for i in range (0,570):
            ruta_save = ruta+"/"+str(categoria)+"/"+str(categoria)+"_"+str(i)+".jpg"
            print(ruta_save)
            img = cv2.imread(ruta_origin_img)
            cv2.imwrite(ruta_save, rotarImagen(img, random.randint(0, 360)))



def save_dataset(ruta, rut_img_ini, limite= 10, pasos_grados= 1,
                 image_x_category = 5, move_diag = 10):
    """
    funcion para guardar el dataset de imagenes rotadas en una carpeta

    :param ruta: ruta donde se guardara el dataset
    :param rut_img_ini:  ruta de las imagenes originales
    :param limite:  limite de imagenes a rotar
    :param pasos_grados:  pasos de grados a rotar
    :return:  None
    """
    for categoria in range (limite):
        print(categoria)
        id_image = 0
        for image_ini_id in range (image_x_category):
            ruta_origin_img = rut_img_ini+"/"+ str(categoria)+"/"+str(categoria)+"_"+str(image_ini_id)+".jpg"
            print(ruta_origin_img)
            #rotaacion de imagenes 0 a 360 grados
            for grados in range (0,360,pasos_grados):
                ruta_save = ruta+"/"+str(categoria)+"/"+str(categoria)+"_"+str(id_image)+".jpg"
                id_image += 1
                print(ruta_save)
                img = cv2.imread(ruta_origin_img)
                cv2.imwrite(ruta_save, rotarImagen(img, grados))

            #traslacion de imagenes de -100 a 100 pixeles en x y y de pasos de 10 pixeles
            for move in range (-100,100,move_diag):
                ruta_save = ruta+"/"+str(categoria)+"/"+str(categoria)+"_"+str(id_image)+".jpg"
                id_image += 1
                print(ruta_save)
                img = cv2.imread(ruta_origin_img)
                img_norm = scale_tamani_gris(img)
                cv2.imwrite(ruta_save, translate_image(img_norm, move, move))





save_dataset_test("dataset/test", "assets/image_test", limite= 10)
# save_dataset(ruta, ruta_origin, limite= 10, pasos_grados= 1)

# Ejemplo de uso
# img = cv2.imread("assets/image_test/1_0.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.resize(img, (128,128))
# cv2.imwrite("assets/rotacion_70.jpg", img)



# Ejemplo de uso
# shift_x = 80
# shift_y = 80
# image1 = cv2.imread("assets/image/0.jpg")
# image_gris = scale_tamani_gris(image1)
# # cv2.imwrite("assets/escala_1.jpg", image_gris)
#
# # image = cv2.resize(image1, (128,128))
# # image3 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("assets/escala_1.jpg", translate_image(image_gris, shift_x, shift_y))

