import tensorflow as tf
from tensorflow.keras.layers import Layer

class CTC(Layer):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, inputs):
        y_pred, labels, input_length, label_length = inputs
        loss = self.loss_fn(labels, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred