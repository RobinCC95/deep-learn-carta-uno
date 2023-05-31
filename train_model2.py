import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
alto = 128
ancho = 128
pixeles = alto * ancho
numero_canales=1
formaImagen=(ancho,alto,numero_canales)
#por ser 10 digitos o 10 clasificacinones
numeroCategorias=10

cantidaDatosEntrenamiento=[1900, 1900, 1900,1900, 1900, 1900, 1900, 1900, 1900, 1900]
cantidaDatosPruebas=[80, 80, 80, 80, 80, 80, 80, 80, 80, 80]


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

#Cargar las imágenes
imagenes, probabilidades=cargarDatos("dataset/train/",numeroCategorias,cantidaDatosEntrenamiento,ancho,alto)


model = Sequential()

# Capa de entrada
model.add(InputLayer(input_shape=(pixeles,)))
model.add(Reshape(formaImagen))
# Capas convolucionales
model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

# Capas totalmente conectadas
model.add(Flatten())
model.add(Dense(128, activation='relu'))

# Capa de salida
model.add(Dense(numeroCategorias, activation='softmax'))

# Compilación y entrenamiento
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=imagenes, y=probabilidades, epochs=40, batch_size=320)

# Evaluación del modelo
imagenesPrueba,probabilidadesPrueba=cargarDatos("dataset/test/",numeroCategorias,cantidaDatosPruebas,ancho,alto)
resultados = model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)
print("Accuracy=", resultados[1])

# Guardar modelo
ruta = "models/modeloB.h5"
model.save(ruta)

# Informe de estructura de la red
model.summary()
