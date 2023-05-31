from tensorflow.python.keras.models import load_model
import numpy as np
import cv2

MODEL_PATH = "models/modeloB.h5"
def predict(image):
    model=load_model(MODEL_PATH)
    load_images=[]

    #procesar la imagen
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    # cv2.imwrite(f"assets/card_{i}.jpg", image)
    image = image.flatten()
    image = image / 255
    load_images.append(image)
    load_images_npa=np.array(load_images)
    predictions=model.predict(x=load_images_npa)
    print("Predicciones = ",predictions)
    high_class=np.argmax(predictions,axis=1)
    return high_class[0]
