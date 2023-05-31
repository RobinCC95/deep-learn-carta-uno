import cv2
import numpy as np


def cargarDatos(rutaOrigen, numeroCategorias, limite, ancho, alto):
    """Función para cargar las imágenes de entrenamiento y pruebas

    Args:
        rutaOrigen (_type_): _description_
        numeroCategorias (_type_): _description_
        limite (_type_): _description_
        ancho (_type_): _description_
        alto (_type_): _description_

    Returns:
        _type_: retorrna un arreglo con las imágenes y otro con las probabilidades
    """
    imagenesCargadas = []
    valorEsperado = []
    for categoria in range(0, numeroCategorias):
        for idImagen in range(0, limite[categoria]):
            ruta = rutaOrigen + str(categoria) + "/" + str(categoria) + "_" + str(idImagen) + ".jpg"
            print(ruta)
            imagen = cv2.imread(ruta)  # cargo imagen
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # convierto a escala de grises
            imagen = cv2.resize(imagen, (ancho, alto))  # redimensiono
            imagen = imagen.flatten()  # aplanar de matriz a vector
            imagen = imagen / 255  # normalizar
            imagenesCargadas.append(imagen)
            probabilidades = np.zeros(
                numeroCategorias)  # creo un vector de 10 posiciones que representa las 10 categorias
            probabilidades[
                categoria] = 1  # asigno la posicion de la categoria que corresponde cero = [1,0,0,0,0,0,0,0,0,0]
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento = np.array(imagenesCargadas)
    valoresEsperados = np.array(valorEsperado)
    return imagenesEntrenamiento, valoresEsperados