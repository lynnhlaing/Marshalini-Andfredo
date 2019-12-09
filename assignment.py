import numpy as np
import tensorflow as tf
import preprocess.py
import tensorflow as tf
from tensorflow.keras import Model

class Inception_Model(tf.keras.Model):

    def __init__(self):
        '''Init hyperparameters and layers'''
        #hyperparameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.batch_size = 100
        self.inception_filters = 10
        #layers
        self.incpetion_1 = tf.keras.layers.Conv2D(filters=self.inception_filters, kernel_size=[12,4], padding='same',activation='relu')
        self.incpetion_2 = tf.keras.layers.Conv2D(filters=self.inception_filters, kernel_size=[14,8], padding='same',activation='relu')
        self.incpetion_3 = tf.keras.layers.Conv2D(filters=self.inception_filters, kernel_size=[16,16], padding='same',activation='relu')

        self.concatenate = tf.keras.layers.concatenate()

        self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size=2)

        self.conv_1 = tf.keras.layers.Conv2D(filters=self.inception_filters, kernel_size=[1,4], padding='valid',activation='relu')

        self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size=2)
        self.max_pool3 = tf.keras.layers.MaxPool2D(pool_size=2)

        self.conv_2 = tf.keras.layers.Conv2D(filters=self.inception_filters/2, kernel_size=[1,1], padding='same',activation='relu')
        self.conv_3 = tf.keras.layers.Conv2D(filters=self.inception_filters/4, kernel_size=[1,1], padding='same',activation='relu')
        self.conv_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=[1,80], padding='same',activation='softmax')



    @tf.function
    def call(self, inputs):
        '''model call'''
        pass

    @tf.function
    def loss_function(self, inputs, labels);
        pass

    def accuracy_function(self, output, labels):
        pass


def train(model, train_inputs, train_labels):
    pass

def test(model, test_inputs, test_labels):
    pass

def main():
    pass

if __name__ == '__main__':
    main()
