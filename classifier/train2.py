import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import cv2

directory = "../misc/dataset5"
classes = np.array(["black", "blue", "brown", "green", "pink", "red", "white", "yellow"])

lengths = []
for c in classes:
    class_dir = directory + "/" + c
    files = os.listdir(class_dir)
    lengths.append(len(files))

lengths = np.array(lengths)
size = np.amin(lengths)

images = []
labels = []
for c in classes:
    class_dir = directory + "/" + c
    files = os.listdir(class_dir)
    random.shuffle(files)
    files = files[:size]

    for f in files:
        #images.append(cv2.imread(class_dir + "/" + f, cv2.IMREAD_COLOR))
        #images.append(cv2.cvtColor(cv2.imread(class_dir + "/" + f, cv2.IMREAD_COLOR), cv2.COLOR_BGR2HSV))
        bgr = cv2.imread(class_dir + "/" + f, cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(cv2.imread(class_dir + "/" + f, cv2.IMREAD_COLOR), cv2.COLOR_BGR2HSV)
        combined = np.append(bgr, hsv, axis=2)
        images.append(combined)
        labels.append(np.where(classes == c))

images = np.array(images)
labels = np.array(labels).reshape(-1)
labels = keras.utils.to_categorical(labels)

shuffler = np.random.permutation(len(labels))
images = images[shuffler]
labels = labels[shuffler]
images = images / 255.0

border = int(labels.shape[0] * 0.8)

train_images = images[:border]
train_labels = labels[:border]
test_images = images[border:]
test_labels = labels[border:]
print(f"Training images:", border)
print(f"Test images:", labels.shape[0] - border)

#print(test_labels)


model = keras.models.Sequential([
    keras.layers.Conv2D(8, (5, 5), strides=(1, 1), activation="relu", input_shape=(18, 18, 6)),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(8)
])

print(model.summary())

loss = keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model.fit(train_images, train_labels, epochs=10, batch_size=5, verbose=1)

model.evaluate(test_images, test_labels, batch_size=5, verbose=2)

plt.plot(history.history["loss"], label="loss")
plt.ylim([0, 2])
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()

model.add(keras.layers.Softmax())

model.save("classifier_combined3.h5")