import cv2
from pynput import keyboard as kb
from cut import cut
from predict import predict
PATH = "assets/main_card.jpg"
def do_nothing_on_press(key):
    pass


def release_p(key):
    if key == kb.KeyCode.from_char('p'):
        cv2.imwrite(PATH, frame)
        img_list = cut(cv2.imread(PATH))
        result_list = []
        for img in img_list:
            result_list.append(predict(img))
        print(result_list)


video = cv2.VideoCapture(1)

listener = kb.Listener(do_nothing_on_press, release_p)
listener.start()

while listener.is_alive():
    _, frame = video.read()
    cv2.imshow("Detección de Cartas", frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
video.release()
cv2.destroyAllWindows()