from pandas.core.common import random_state

from ImagePreProcessing import MasterImage
from CNNModel import model, class_names, test_acc, test_loss, X_Data, Y_Data
import matplotlib.pyplot as plt
import numpy as np
from CNNModel import history


plt.plot(test_loss, test_acc, "b", label="trainning accuracy")

plt.legend()
plt.show()


# Draft testing: Recall
"""from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

classifier = make_pipeline(StandardScaler(), LinearSVC(random_state=random_state))
classifier.fit(X_Data, Y_Data)"""


