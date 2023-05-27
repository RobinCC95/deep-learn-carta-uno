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

cantidaDatosEntrenamiento=[72,72, 72, 72, 72, 72, 72, 72, 72, 72]
cantidaDatosPruebas=[20,20,20,20,20,20,20,20,20,20]

#Capa entrada
model=Sequential()
#capa de entrada de 784 neuronas
model.add(InputLayer(input_shape=(pixeles,)))
#convierto de nuevo a matriz
model.add(Reshape(formaImagen))

#Capas Ocultas
#Capas convolucionales

#kernerl_size=5,5 --> tamaño de la ventana o filtro
#strides=2,2 --> tamaño del paso
#padding="same" --> relleno al final de la imagen, same -> duplica las ultimas filas y columnas
model.add(Conv2D(kernel_size=5,strides=2,filters=16,padding="same",activation="relu",name="capa_1"))
#pool_size=2,2 --> tamaño de la ventana o filtro reducido que obtiene los datos mas relevantes de la imagen
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(kernel_size=3,strides=1,filters=36,padding="same",activation="relu",name="capa_2"))
model.add(MaxPool2D(pool_size=2,strides=2))

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
model.fit(x=imagenes,y=probabilidades,epochs=30,batch_size=60)
