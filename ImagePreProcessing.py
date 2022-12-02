import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import cv2
from keras.preprocessing.image import ImageDataGenerator


class MasterImage(object):
    def __init__(self, PATH='', IMAGE_SIZE=256):
        self.PATH = PATH
        self.IMAGE_SIZE = IMAGE_SIZE
        self.image_data = []
        self.x_data = []
        self.y_data = []
        self.CATEGORIES = []

        # This will get List of categories
        self.list_categories = []

    def get_categories(self):
        for path in os.listdir(self.PATH):
            self.list_categories.append(path)
        print("Found Categories ", self.list_categories, '\n')
        return self.list_categories

    def Process_Image(self):
        try:
            """
            Return Numpy array of image
            :return: X_Data, Y_Data
            """
            self.CATEGORIES = self.get_categories()
            for categories in self.CATEGORIES:  # Iterate over categories

                train_folder_path = os.path.join(self.PATH, categories)  # Folder Path
                class_index = self.CATEGORIES.index(categories)  # this will get index for classification

                for img in os.listdir(train_folder_path):  # This will iterate in the Folder
                    new_path = os.path.join(train_folder_path, img)  # image Path

                    try:  # if any image is corrupted
                        image_data_temp = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)  # Read Image as numbers
                        image_temp_resize = cv2.resize(image_data_temp, (self.IMAGE_SIZE, self.IMAGE_SIZE))
                        self.image_data.append([image_temp_resize, class_index])
                    except:
                        pass

            data = np.asanyarray(self.image_data)

            # Iterate over the Data
            for x in data:
                self.x_data.append(x[0])  # Get the X_Data
                self.y_data.append(x[1])  # get the label

            X_Data = np.asarray(self.x_data) / (255.0)  # Normalize Data
            Y_Data = np.asarray(self.y_data)

            # reshape x_Data

            X_Data = X_Data.reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1)

            return X_Data, Y_Data
        except:
            print("Failed to run Function Process Image ")

    def pickle_image(self):

        """
        :return: None Creates a Pickle Object of DataSet
        """
        if 'train' in self.PATH:
            # Call the Function and Get the Data
            X_Data, Y_Data = self.Process_Image()

            # Write the Entire Data into a Pickle File
            pickle_out = open('X_Data', 'wb')
            pickle.dump(X_Data, pickle_out)
            pickle_out.close()

            # Write the Y Label Data
            pickle_out = open('Y_Data', 'wb')
            pickle.dump(Y_Data, pickle_out)
            pickle_out.close()

            print("Pickled Image Successfully ")
            return X_Data, Y_Data

        elif 'test' in self.PATH:
            # Call the Function and Get the Data
            X_Test, Y_Test = self.Process_Image()

            # Write the Entire Data into a Pickle File
            pickle_out = open('X_Test', 'wb')
            pickle.dump(X_Test, pickle_out)
            pickle_out.close()

            # Write the Y Label Data
            pickle_out = open('Y_Test', 'wb')
            pickle.dump(Y_Test, pickle_out)
            pickle_out.close()

            print("Pickled Image Successfully ")
            return X_Test, Y_Test



    def load_dataset(self):

        if 'train' in self.PATH:
            try:
                # Read the Data from Pickle Object
                X_Temp = open('X_Data', 'rb')
                X_Data = pickle.load(X_Temp)
                Y_Temp = open('Y_Data', 'rb')
                Y_Data = pickle.load(Y_Temp)

                print('Reading Dataset from PIckle Object')

                return X_Data, Y_Data

            except:
                print('Could not Found Pickle File ')
                print('Loading File and Dataset  ..........')

                X_Data, Y_Data = self.pickle_image()
                return X_Data, Y_Data

        elif 'test' in self.PATH:
            try:
                # Read the Data from Pickle Object
                X_Temp = open('X_Test', 'rb')
                X_Test = pickle.load(X_Temp)
                Y_Temp = open('Y_Test', 'rb')
                Y_Test = pickle.load(Y_Temp)

                print('Reading Dataset from PIckle Object')

                return X_Test, Y_Test

            except:
                print('Could not Found Pickle File ')
                print('Loading File and Dataset  ..........')

                X_Test, Y_Test = self.pickle_image()
                return X_Test, Y_Test










