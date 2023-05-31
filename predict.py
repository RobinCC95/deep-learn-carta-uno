from tensorflow.python.keras.models import load_model
import numpy as np
import cv2

MODEL_PATH = "models/modeloA.h5"
def predict(image):
    model=load_model(MODEL_PATH)
    load_images=[]
    load_images.append(image)
    load_images_npa=np.array(load_images)
    predictions=model.predict(x=load_images_npa)
    print("Predicciones = ",predictions)
    high_class=np.argmax(predictions,axis=1)
    return high_class[0]
