import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ImagePreProcessing import MasterImage

path = r'crab_raw_data/data/train'
crabs = MasterImage(PATH=path,
                    IMAGE_SIZE=244)
X_Data, Y_Data = crabs.load_dataset()
class_names = crabs.get_categories()
#print(class_names)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(80,80,1)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_Data, Y_Data, epochs=20)
