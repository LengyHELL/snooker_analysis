import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import cv2


dir = "./misc/dataset"
file = open(dir + "/info.txt")
files = file.read().strip().split("\n")
file.close()

random.shuffle(files)

labels = []
images = []
for f in files:
    array = np.load(dir + "/labels/" + f + ".npy")
    #array = array / 2
    array = array[0]
    labels.append(array.flatten())
    image = cv2.imread(dir + "/images/" + f + ".png")
    #image = cv2.resize(image, (512, 256))
    images.append(image / 255)

labels = np.array(labels)
images = np.array(images)

print(labels[0])
border = int(labels.shape[0] * 0.8)

train_images = images[:border]
train_labels = labels[:border]
test_images = images[border:]
test_labels = labels[border:]
print("Training images:", border)
print("Test images:", labels.shape[0] - border)

model = keras.models.Sequential([
    keras.layers.Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation="relu", input_shape=(512, 1024, 3)),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(1, (3, 3), strides=(1, 1), padding="same", activation="relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(176, activation="relu"),
    keras.layers.Dense(176, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(88, activation="relu"),
    keras.layers.Dense(2, activation="relu")
])

print(model.summary())

loss = keras.losses.MeanAbsoluteError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model.fit(train_images, train_labels, epochs=10, batch_size=8, verbose=1)

model.evaluate(test_images, test_labels, batch_size=1, verbose=2)

out = model.predict(test_images[:1])
out = np.int0(np.reshape(out, (-1, 2)))
image = np.int0(test_images[0] * 255).astype("uint8")
#print(out)

for p in out:
    p = (p[0] - 10, p[1] - 10)
    cv2.rectangle(image, p, (p[0] + 20, p[1] + 20), (0, 0, 255), 1)

for p in test_labels[:1]:
    p = (p[0] - 10, p[1] - 10)
    cv2.rectangle(image, p, (p[0] + 20, p[1] + 20), (255, 0, 0), 1)

plt.figure(figsize=(16, 6))

plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(122)
plt.plot(history.history["loss"], label="loss")
plt.ylim([0, 250])
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()

#model.save("recognition_model.h5")
