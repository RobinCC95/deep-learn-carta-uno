from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Cargar el modelo desde el archivo h5
ruta_modelo = "models/modeloB.h5"
model = load_model(ruta_modelo)

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
