import random
import datetime
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from CustomMetrics import MetricsCallback

from ImagePreProcessing import MasterImage
from keras.preprocessing.image import ImageDataGenerator

path = r'crab_raw_data/data/train'
crabs = MasterImage(PATH=path,
                        IMAGE_SIZE=256)
X_Data, Y_Data = crabs.load_dataset()
class_names = crabs.get_categories()

path = r'crab_raw_data/data/test'
testcrabs = MasterImage(PATH=path,
                        IMAGE_SIZE=256)
X_Test, Y_Test = testcrabs.load_dataset()

    # augment the images
train_generator = ImageDataGenerator(rescale=1 / 255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)

# CNN Model
model = keras.Sequential([
    keras.layers.Input(shape=(256, 256 , 1)), # 1 is the num_channel because of grey scale

    # First Conv-Relu-MaxPool Layer
    keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=16, strides=8),

    # Second Conv-Relu-MaxPool Layer
    keras.layers.Conv2D(filters=12, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=4, strides=4),

    # Third Conv-Relu-MaxPool Layer
    keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=4, strides=4),



    keras.layers.Flatten(),
    keras.layers.Dense(units=4, activation='relu'),
    keras.layers.Dense(units=2, activation="softmax")
    ]
)

model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
metrics_callback = MetricsCallback(test_data = X_Test, y_true = Y_Test)
history = model.fit(X_Data, Y_Data, epochs=25, batch_size=3, callbacks=[metrics_callback, tensorboard_callback])
test_loss, test_acc = model.evaluate(X_Data, Y_Data)

# Manually testing model
print('\nTest accuracy:', test_acc)
tf.keras.models.save_model(filepath='Model/',model=model)


# draft testing model. A little mess need to be organize :))
print(Y_Test)
predictions = model.predict(X_Test)
print(str(predictions.shape))

"""count = 0
while True:
    wrong = 0
    i = random.randint(0,300)
    plt.figure(figsize=(5,5))
    plt.imshow(X_Test[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[Y_Test[i]])
    plt.title(class_names[np.argmax(predictions[i])])
    plt.show()
    count += 1
    if (str(class_names[Y_Test[i]]) != str(class_names[np.argmax(predictions[i])])):
        wrong += 1
    if wrong == 10:
        break

print(count)"""
