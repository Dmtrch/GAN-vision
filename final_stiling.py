# Импорт необходимых библиотек
import cv2
from keras.models import load_model
import numpy as np

# Загрузка моделей нейронных сетей
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gan_model = load_model('gan_model.h5')

# Загрузка исходного изображения
img = cv2.imread('original_photo.jpg')

# Обнаружение лица
face = face_detector.detectMultiScale(img)[0]
x, y, w, h = face

# Выделение области лица
face_img = img[y:y+h, x:x+w]

# Нормализация размера лица
face_img = cv2.resize(face_img, (224, 224))

# Предсказание стилизованного лица GAN
styled_face = gan_model.predict(np.expand_dims(face_img, 0))[0]
styled_face = cv2.resize(styled_face, (w, h))

# Вставка стилизованного лица в исходное изображение
img[y:y+h, x:x+w] = styled_face

# Вывод результата
cv2.imwrite('styled_img.jpg', img)
cv2.imshow('result', img)
cv2.waitKey()