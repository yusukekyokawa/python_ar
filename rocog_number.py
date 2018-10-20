import cv2
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import load_model


def preprocessing(img):
    img = img[190:290,270:370]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (28, 28))
    res, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = 255 - img
    img = img.astype(np.float32)
    cv2.imwrite("img.jpg",img)
    img /= 255
    img = np.array(img).reshape(1,784)
    return img


def main():
    # 学習済みモデルの読み込み
    model = load_model('my_model.h5')
    # Webカメラの映像表示
    capture = cv2.VideoCapture(0)
    if capture.isOpened() is False:
            raise("IO Error")
    while True:
        # Webカメラの映像とりこみ
        ret, image = capture.read()
        if ret == False:
            continue
        # Webカメラの映像表示
        cv2.rectangle(image,(270,190),(370,290),(0,0,255),3)
        cv2.imshow("Capture", image)
        k = cv2.waitKey(10)
        # Eキーで処理実行
        if k == 101:
            img = preprocessing(image)
            num = model.predict(img)
            # cv2.imwrite("img.jpg",img)
            print(num.data)
            print(np.argmax(num.data))
        # ESCキーでキャプチャー画面を閉じる
        if k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()