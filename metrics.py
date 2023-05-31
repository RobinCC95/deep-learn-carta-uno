from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix
from carga_datos import cargarDatos
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Cargar el modelo desde el archivo h5
ruta_modelo = "models/modeloC.h5"
model = load_model(ruta_modelo)
cantidaDatosPruebas=[560, 560, 560, 560, 560, 560, 560, 560, 560, 560]
numeroCategorias=10
ancho = 128
alto = 128
# Cargar los datos de prueba
imagenesPrueba, probabilidadesPrueba = cargarDatos("dataset/test/", numeroCategorias, cantidaDatosPruebas, ancho, alto)

# Obtener las predicciones del modelo
predicciones = model.predict(imagenesPrueba)

# Convertir las probabilidades en etiquetas predichas
etiquetasPredichas = np.argmax(predicciones, axis=1)
etiquetasReales = np.argmax(probabilidadesPrueba, axis=1)

# Calcular y mostrar el accuracy
accuracy = accuracy_score(etiquetasReales, etiquetasPredichas)
print("Accuracy:", accuracy)

# Calcular y mostrar la precision
precision = precision_score(etiquetasReales, etiquetasPredichas, average='macro')
print("Precision:", precision)

# Calcular y mostrar el recall
recall = recall_score(etiquetasReales, etiquetasPredichas, average='macro')
print("Recall:", recall)

# Calcular y mostrar el F1 Score
f1 = f1_score(etiquetasReales, etiquetasPredichas, average='macro')
print("F1 Score:", f1)

# Calcular y mostrar la matriz de confusión
matriz_confusion = confusion_matrix(etiquetasReales, etiquetasPredichas)
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Etiquetas Predichas')
plt.ylabel('Etiquetas Reales')
plt.show()