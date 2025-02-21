import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.layers import Input, Conv2D, concatenate, GaussianNoise, Flatten, Dense
from keras.models import Model, Sequential
from keras.preprocessing import image
import keras.backend as K
from tqdm import tqdm

# Constants
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
IMG_SHAPE = (64, 64)

# Load dataset
def load_dataset_small(num_images_per_class_train=10):
    X_train = []
    for c in os.listdir(TRAIN_DIR):
        c_dir = os.path.join(TRAIN_DIR, c, 'images')
        c_imgs = os.listdir(c_dir)
        random.shuffle(c_imgs)
        for img_name in c_imgs[:num_images_per_class_train]:
            img = image.load_img(os.path.join(c_dir, img_name))
            x = image.img_to_array(img)
            X_train.append(x)
    return np.array(X_train) / 255.0

X_train = load_dataset_small()

print("Number of training examples =", X_train.shape[0])
print("X_train shape:", X_train.shape)

# Splitting dataset
input_S1 = X_train[0:X_train.shape[0] // 4]
input_S2 = X_train[X_train.shape[0] // 4 : 2*(X_train.shape[0] // 4)]
input_S3 = X_train[2*(X_train.shape[0] // 4) : 3*(X_train.shape[0] // 4)]
input_C = X_train[3*(X_train.shape[0] // 4):]

# Generate Stego Images
stego_images = encoder_model.predict([input_S1, input_S2, input_S3, input_C])

# Create Labels (0 = Normal, 1 = Stego)
y_labels = np.concatenate([np.zeros(len(input_C)), np.ones(len(stego_images))])
X_combined = np.concatenate([input_C, stego_images])

# Define Steganalysis Classifier
classifier = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE + (3,)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output
])

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_combined, y_labels, epochs=5, batch_size=32)

# Test Steganalysis on a New Image
def detect_steganography(image_path):
    new_image = image.load_img(image_path, target_size=IMG_SHAPE)
    new_image = np.array(image.img_to_array(new_image)) / 255.0
    new_image = new_image.reshape(1, *IMG_SHAPE, 3)
    
    prediction = classifier.predict(new_image)
    if prediction > 0.5:
        print("🔴 This image likely contains hidden information (Stego).")
    else:
        print("🟢 This image is normal (No Stego detected).")

# Example Usage
detect_steganography('test_image.jpg')