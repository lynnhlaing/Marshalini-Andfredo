import numpy as np
import tensorflow as tf
import preprocess
import tensorflow as tf
from tensorflow.keras import Model

class Inception_Model(tf.keras.Model):

    def __init__(self):
        '''Init hyperparameters and layers'''
        #hyperparameters

        self.batch_size = 128
        self.inception_filters = 36
        #layers
        self.inception_1 = tf.keras.layers.Conv2D(filters=self.inception_filters, kernel_size=[12,4], padding='same',activation='relu')
        self.inception_2 = tf.keras.layers.Conv2D(filters=self.inception_filters, kernel_size=[14,8], padding='same',activation='relu')
        self.inception_3 = tf.keras.layers.Conv2D(filters=self.inception_filters, kernel_size=[16,16], padding='same',activation='relu')

        self.concatenate = tf.keras.layers.concatenate(axis=3)

        self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size=2)
        self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size=2)

        self.conv_1 = tf.keras.layers.Conv2D(filters=self.inception_filters, kernel_size=[1,4], padding='same',activation='relu')

        self.max_pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,3))
        self.max_pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,3))

        self.conv_2 = tf.keras.layers.Conv2D(filters=self.inception_filters/2, kernel_size=[1,1], padding='same',activation='relu')
        self.conv_3 = tf.keras.layers.Conv2D(filters=self.inception_filters/4, kernel_size=[1,1], padding='same',activation='relu')
        self.conv_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=[1,1], padding='same',actvation="softmax")

        #self.dense = tf.keras.layers.Dense(units = 2)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    @tf.function
    def call(self, inputs):
        '''runs the model architecture and outputs the classification logits
        input = MFCC encoded timbre data for 1024 segments of each song each having 12 features (batch_size,1024,12)
        output = dancibility prediction value
        '''
        inception1 = self.inception_1(inputs)
        inception2 = self.inception_2(inputs)
        inception3 = self.inception_3(inputs)
        combined = self.concatenate((inception1,inception2,inception3))

        max1 = self.max_pool2(self.max_pool1(combined))

        conv1 = self.conv_1(max1)

        max2 = self.max_pool4(self.max_pool3(conv1))

        out = self.conv_4(self.conv_3(self.conv_2(max2)))

        #return tf.dense(out)
        return reduce_mean(out)

    def create_classes(self, probs):
        return tf.greater(probs,0.5)

    @tf.function
    def loss_function(self, probs, labels):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels,self.create_classes(probs)))

    def accuracy_function(self, probs, labels):
        assignments = self.create_classes(probs)
        correct = 0
        acc = 0
        for song in assignments:
            if song == labels[acc]:
                correct+=1
            acc+=1
        return correct/assignments.shape[0]


def create_kfolds(data, labels, folds=10):
    assert(data.shape(0) == labels.shape(0))
    #7400 --> 10 x 740 7400, 1024 x 12
    data = data[0:7400]
    labels = labels[0:7400]
    batched_data = np.reshape(data, (10,740,1024,12))
    batched_labels = np.reshape(labels, (10,740,1))

    return batched_data, batched_labels



def train(model, train_inputs, train_labels):
    num_batches = (train_inputs.shape)[0]
    for i in range(num_batches):
        batch_inputs = np.squeeze(train_inputs[i,:,:],axis=0)
        batch_labels = np.squeeze(train_labels[i,:],axis=0)
        with tf.GradientTape() as tape:
            probs = model.call(batch_inputs)
            loss = model.loss_function(probs,batch_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients,model.trainable_variables))



def test(model, test_inputs, test_labels):
    probs = model.call(test_inputs)
    accuracy = model.accuracy_function(probs,test_labels)
    print(accuracy)

def main():
    data, labels = preprocess.get_data("./MillionSongSubset/")
    # model = Inception_Model()
    # num_epochs = 5
    # while i <= num_epochs:
    #     num_songs = data.shape[0]
    # 	idx = np.arange(num_songs)
    # 	shuffled = tf.random.shuffle(idx)
    # 	shuffled_inputs = tf.gather(data, shuffled, axis=0)
    # 	shuffled_labels = tf.gather(labels, shuffled, axis=0)
    #     batched_data, batched_labels = create_kfolds(data, labels)
    #     for i in range(0, batched_data.shape(0)):
    #         if i == 0:
    #             train_data = batched_data[1:batched_data.shape(0)]
    #             train_labels = batched_labels[1:batched_data.shape(0)]
    #             test_data = batched_data[0,:,:,:]
    #             test_labels = batched_labels[0,:,:]
    #         else:
    #             train_data = np.stack((batched_data[:i],batched_data[i:]),axis=0)
    #             train_labels = np.stack((batched_labels[:i],batched_labels[i:]),axis=0)
    #             test_data = batched_data[i]
    #             test_labels = batched_labels[i]
    #
    #         train(model,train_data,train_labels)
    #         test(model,test_data,test_labels)



if __name__ == '__main__':
    main()
