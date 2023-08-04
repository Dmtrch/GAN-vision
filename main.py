import cv2
import numpy as np
import torch
import torchvision

# Инициализация Metal backend
torch.backends.mps.is_available()
torch.backends.mps.is_built()

# Выбор устройства
device = torch.device('cpu')

# Загрузка изображений
img1 = cv2.imread('face1.jpg')
img2 = cv2.imread('face2.jpg')

# Обнаружение лица
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face = face_detector.detectMultiScale(img1, 1.3, 5)
x, y, w, h = face[0]
face_img = img1[y:y+h, x:x+w]
##cv2.imshow('face', face_img)
##cv2.waitKey()

# Загрузка и инициализация модели
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
model = model.eval().to(device)

# Предсказание маски лица
face_img_t = torch.from_numpy(face_img).permute(2,0,1).unsqueeze(0).to(device).float()
out = model(face_img_t)['out']
face_mask = out.argmax(0).byte().cpu().numpy()

# Масштабирование маски
face_mask = cv2.resize(face_mask, (w, h))

# Наложение лица на фон с учетом индексации
## bg_img[x:x+w, y:y+h, 0][face_mask == 1] = face_img[face_mask == 1]
bg_img = img2.copy()
bg_img[y:y+h, x:x+w] = face_img


# Отображение и сохранение
cv2.imshow('Result', bg_img)
cv2.imwrite('result.jpg', bg_img)
cv2.waitKey()