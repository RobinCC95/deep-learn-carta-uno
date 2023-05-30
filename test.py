
import cv2
from Prediccion import  Prediccion

clases=["Carta 0", "Carta 1", "Carta 2", "Carta 3", "Carta 4", "Carta 5","Carta 6", "Carta 7", "Carta 8", "Carta 9"]

ancho=128
alto=128

miModeloCNN=Prediccion("models/modeloA.h5",ancho,alto)
# id_carta = input("Ingrese el id de la carta del 0 al 9: ")
# id_numero = input("Ingrese el id del numero de la carta del 0 al 79: ")
# imagen=cv2.imread("dataset/test/"+id_carta+"/"+str(id_carta)+"_"+str(id_numero)+".jpg")
imagen=cv2.imread("dataset/test/7/7_1.jpg")

claseResultado=miModeloCNN.predecir(imagen)
print("La imagen cargada es ",clases[claseResultado])

while True:
    cv2.imshow("imagen",imagen)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cv2.destroyAllWindows()
