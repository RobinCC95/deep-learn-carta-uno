import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
from carga_datos import cargarDatos
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


#Capa entrada
model=Sequential()
#capa de entrada de 784 neuronas
model.add(InputLayer(input_shape=(pixeles,)))
#convierto de nuevo a matriz
model.add(Reshape(formaImagen))

#Capas Ocultas
#Capas convolucionales

# Capa 1
model.add(Conv2D(kernel_size=3, strides=1, filters=32, padding="same", activation="relu", name="capa_1"))
model.add(MaxPool2D(pool_size=2, strides=2))

# Capa 2
model.add(Conv2D(kernel_size=3, strides=1, filters=64, padding="same", activation="relu", name="capa_2"))
model.add(MaxPool2D(pool_size=2, strides=2))



#Aplanamiento
model.add(Flatten())
model.add(Dense(128,activation="relu"))

#Capa de salida
model.add(Dense(numeroCategorias,activation="softmax"))

#Traducir de keras a tensorflow
model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=["accuracy"])
#Entrenamiento
#epochs=30 --> cantidad de iteraciones
#batch_size=60 --> cantidad de datos que se van a procesar en cada iteracion
model.fit(x=imagenes,y=probabilidades,epochs=20,batch_size=300)


#Prueba del modelo
imagenesPrueba,probabilidadesPrueba=cargarDatos("dataset/test/",numeroCategorias,cantidaDatosPruebas,ancho,alto)
resultados=model.evaluate(x=imagenesPrueba,y=probabilidadesPrueba)
print("Accuracy=",resultados[1])

# Guardar modelo
ruta="models/modeloA.h5"
model.save(ruta)
# Informe de estructura de la red
model.summary()