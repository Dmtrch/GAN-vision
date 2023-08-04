import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
## import torch.nn.functional as Fimport torch.nn.functional as F

# Загрузка модели
model = torch.hub.load('pytorch/vision:v0.10.0',
                       'deeplabv3_resnet50', pretrained=True)
model.eval()

# Чтение и преобразование изображения
img = cv2.imread('face4.jpeg')
##plt.imshow(img)
##plt.show()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
##plt.imshow(img)
##plt.show()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.ToTensor()
])

img = transform(img)

img = img.unsqueeze(0)


with torch.no_grad():
    output = model(img)['out']

    output = output.squeeze(0).cpu().numpy()  # Преобразуем в NumPy массив

# Преобразование результата в изображение с использованием цветовой маппингов
colors = model.classifier[4].weight.detach().cpu().numpy()  # Получаем цветовую маппинг таблицу из модели
colored_mask = colors[np.argmax(output, axis=0)]  # Применяем цветовую маппинг таблицу к предсказанной маске

# Масштабирование значения пикселей в диапазон от 0 до 255
colored_mask = (colored_mask * 255).astype(np.uint8)

# Преобразование тензора в формат изображения
output_image = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)

# Сохранение результата
cv2.imwrite('result.jpg', output_image)

# Отображение
plt.imshow(output_image)
plt.axis('off')  # Отключение осей координат
plt.show()






