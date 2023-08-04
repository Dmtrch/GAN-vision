import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array


from google_images_download import google_images_download

# Константы
IMG_SHAPE = (224, 224, 3)
NOISE_SHAPE = 128
EPOCHS = 100
BATCH_SIZE = 32

# Загрузка данных

downloader = google_images_download.googleimagesdownload()
##downloader = GoogleImagesDownloader()
downloader.download({
    'keywords': 'faces',
    '--chromedriver': '/chromedriver-mac-arm64/chromedriver',
    'limit': 10000,
    'print_paths': False,
    'size': 'medium',
    'aspect_ratio': 'tall',
})

image_paths = downloader.paths[:10000]
images = np.array([
    img_to_array(load_img(path)) for path in image_paths
])
images = (images - 127.5) / 127.5  # Нормализация

# Генератор
generator_input = Input(shape=(NOISE_SHAPE,))
x = Dense(256, activation='relu')(generator_input)
x = Dense(512, activation='relu')(x)
x = Dense(np.prod(IMG_SHAPE), activation='tanh')(x)
generator_output = Reshape(IMG_SHAPE)(x)

generator = Model(generator_input, generator_output)

# Дискриминатор
discriminator_input = Input(shape=IMG_SHAPE)
x = Flatten()(discriminator_input)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

discriminator = Model(discriminator_input, x)

# GAN
gan_input = Input(shape=(NOISE_SHAPE,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

# Компиляция
discriminator_optimizer = Adam(lr=0.0002, beta_1=0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer)

gan_optimizer = Adam(lr=0.0002, beta_1=0.5)
gan.compile(loss='binary_crossentropy', optimizer=gan_optimizer)

# Тренировка
for epoch in range(EPOCHS):
    for batch in range(len(images) // BATCH_SIZE):
        # Получение пачки реальных изображений
        real_images = images[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE]

        # Генерация фейковых изображений
        noise = tf.random.normal(shape=(BATCH_SIZE, NOISE_SHAPE))
        fake_images = generator(noise)

        # Тренировка дискриминатора
        combined_images = tf.concat([real_images, fake_images], axis=0)
        labels = tf.concat([tf.ones((BATCH_SIZE, 1)), tf.zeros((BATCH_SIZE, 1))], axis=0)
        d_loss = discriminator.train_on_batch(combined_images, labels)

        # Тренировка генератора
        noise = tf.random.normal(shape=(BATCH_SIZE, NOISE_SHAPE))
        misleading_targets = tf.ones((BATCH_SIZE, 1))
        g_loss = gan.train_on_batch(noise, misleading_targets)

# Сохранение обученной GAN
gan.save('face_gan.h5')