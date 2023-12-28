import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
from skimage.io import imread, imshow
from skimage.transform import resize
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dropout, Lambda
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
# Don't Show Warning Messages
import warnings
warnings.filterwarnings('ignore')

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1

NUM_TEST_IMAGES = 10

# Data paths
train_data_dir = '/content/drive/MyDrive/MobileNet/Dataset/train'
val_data_dir = '/content/drive/MyDrive/MobileNet/Dataset/val'
test_data_dir = '/content/drive/MyDrive/MobileNet/Dataset/test'

# Define data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def gaussian_noise(image, noise_factor=0.5):
    """
    Add random noise to the image.
    """
    row, col, _ = image.shape
    gauss = np.random.normal(0, noise_factor, (row, col, 1))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 1)
    return noisy

def load_and_preprocess_data(data_dir, augment=False, add_noise=False):
    images = []
    masks = []

    for image_filename in os.listdir(os.path.join(data_dir, 'images')):
        image_path = os.path.join(data_dir, 'images', image_filename)
        mask_path = os.path.join(data_dir, 'masks', image_filename)

        # Read and preprocess image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        image = image / 255.0
        image = np.expand_dims(image, axis=-1)  # Add channel dimension

        # Read and preprocess mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask >= 10).astype(np.float32)  # Ensure the correct data type
        mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
        mask = mask / 255.0  # Normalize to [0, 1]
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension

        # Apply noise if specified
        if add_noise:
            image = gaussian_noise(image)

        images.append(image)
        masks.append(mask)

        # Apply data augmentation if specified
        if augment:
            seed = np.random.randint(1, 1000)
            image = datagen.random_transform(image, seed=seed)
            mask = datagen.random_transform(mask, seed=seed)

            # Ensure the values are still in the [0, 1] range
            image = np.clip(image, 0, 1)
            mask = np.clip(mask, 0, 1)

            images.append(image)
            masks.append(mask)

    return np.array(images), np.array(masks)

# Load and preprocess training data with augmentation and noise
train_images, train_masks = load_and_preprocess_data(train_data_dir, augment=True, add_noise=True)

# Load and preprocess validation data
val_images, val_masks = load_and_preprocess_data(val_data_dir)

# Load and preprocess test data
test_images, test_masks = load_and_preprocess_data(test_data_dir)

# Visualize some training data
# visualize_data(train_images, train_masks)


inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()
filepath = "model.h5"

# earlystopper = EarlyStopping(patience=5, verbose=1)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')

# callbacks_list = [earlystopper, checkpoint]
callbacks_list = [checkpoint]
history = model.fit(train_images, train_masks, validation_data = (val_images,val_masks), batch_size=16, epochs=1000,
                    callbacks=callbacks_list)
