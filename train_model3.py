import tensorflow as tf
import keras
import numpy as np
import cv2
from carga_datos import cargarDatos
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


#Cargar las imágenes
imagenes, probabilidades=cargarDatos("dataset/train/",numeroCategorias,cantidaDatosEntrenamiento,ancho,alto)


model = Sequential()

# Capa de entrada
model.add(InputLayer(input_shape=(pixeles,)))
model.add(Reshape(formaImagen))
# Capas convolucionales
model.add(Conv2D(kernel_size=5,strides=2,filters=16,padding="same",activation="relu",name="capa_1"))
#pool_size=2,2 --> tamaño de la ventana o filtro reducido que obtiene los datos mas relevantes de la imagen
model.add(MaxPool2D(pool_size=2,strides=2))
#capa 2
model.add(Conv2D(kernel_size=3,strides=1,filters=36,padding="same",activation="relu",name="capa_2"))
model.add(MaxPool2D(pool_size=2,strides=2))


# Capas totalmente conectadas
model.add(Flatten())
model.add(Dense(128, activation='relu'))

# Capa de salida
model.add(Dense(numeroCategorias, activation='softmax'))

# Compilación y entrenamiento
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=imagenes, y=probabilidades, epochs=30, batch_size=380)

# Evaluación del modelo
imagenesPrueba,probabilidadesPrueba=cargarDatos("dataset/test/",numeroCategorias,cantidaDatosPruebas,ancho,alto)
resultados = model.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)
print("Accuracy=", resultados[1])

# Guardar modelo
ruta = "models/modeloC.h5"
model.save(ruta)

# Informe de estructura de la red
model.summary()
