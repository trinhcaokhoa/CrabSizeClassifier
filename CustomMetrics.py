import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import classification_report


class MetricsCallback(Callback):
    def __init__(self, test_data, y_true):
        # Should be the label encoding of your classes
        self.y_true = y_true
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        # Here we get the probabilities
        y_pred = self.model.predict(self.test_data)
        # Here we get the actual classes
        y_pred = tf.argmax(y_pred, axis=1)
        # Actual dictionary
        report_dictionary = classification_report(self.y_true, y_pred, output_dict=True)
        # Only printing the report
        print(classification_report(self.y_true, y_pred, output_dict=False))