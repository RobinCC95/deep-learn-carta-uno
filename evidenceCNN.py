import cv2
from pynput import keyboard as kb
from recorte import recortar_imagen as cut
PATH = f"assets/carta_test.jpg"


def do_nothing_on_press(tecla):
    pass


def soltar_p(tecla):
    if tecla == kb.KeyCode.from_char('p'):
        cv2.imwrite(PATH, frame)
        cut(cv2.imread(PATH))
        pass
    pass

video = cv2.VideoCapture(1)

listener = kb.Listener(do_nothing_on_press, soltar_p)
listener.start()

while listener.is_alive():
    _, frame = video.read()
    cv2.imshow("Imagen", frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
