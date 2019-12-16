import numpy as np
import tensorflow as tf
import preprocess
import tensorflow as tf
from tensorflow.keras import Model
import sys

class Inception_Model(tf.keras.Model):

    def __init__(self):
        '''Init hyperparameters and layers'''
        #hyperparameters
        super(Inception_Model, self).__init__()
        self.batch_size = 128
        self.inception_filters = 36
        #layers
        self.inception_1 = tf.keras.layers.Conv2D(filters=self.inception_filters, kernel_size=[12,4], padding='same')
        self.inception_2 = tf.keras.layers.Conv2D(filters=self.inception_filters, kernel_size=[14,8], padding='same')
        self.inception_3 = tf.keras.layers.Conv2D(filters=self.inception_filters, kernel_size=[16,16], padding='same')



        self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size=2)
        self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size=2)

        self.conv_1 = tf.keras.layers.Conv2D(filters=self.inception_filters, kernel_size=[1,4], padding='same', activation='sigmoid')

        self.max_pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,3))
        self.max_pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,3))

        self.conv_2 = tf.keras.layers.Conv2D(filters=int(self.inception_filters/2), kernel_size=[1,1], padding='same')
        self.conv_3 = tf.keras.layers.Conv2D(filters=int(self.inception_filters/4), kernel_size=[1,1], padding='same')
        self.conv_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=[1,1], padding='same')

        # self.dense = tf.keras.layers.Dense(units = 2,activation='softmax')

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
        #[samples,1024,12,1]

        combined = tf.concat((inception1,inception2,inception3),axis=2)
        #[samples,1024,12,3]

        max1 = self.max_pool2(self.max_pool1(combined))
        conv1 = self.conv_1(max1)

        max2 = self.max_pool4(self.max_pool3(conv1))
        conv2 = self.conv_2(max2)
        conv3=self.conv_3(conv2)
        out= tf.nn.sigmoid(tf.squeeze(self.conv_4(conv3)))
        # out = self.conv_4(self.conv_3(self.conv_2(intermediate)))
        #return tf.dense(out)
        # print(out)
        return tf.math.reduce_mean(out, axis=1)

    def create_classes(self, probs):
        return tf.where(probs>0.5,1.0,0.0)


    def loss_function(self, probs, labels):
        assignment = self.create_classes(probs)
        # print(probs)
        return tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(labels, probs))

    def accuracy_function(self, probs, labels):
        assignments = self.create_classes(probs).numpy()
        correct = 0
        acc = 0
        for song in assignments:
            if song == labels[acc]:
                correct+=1
            acc+=1
        return correct/assignments.shape[0]


def create_batched_kfolds(data, labels, folds,number_of_batches):
    assert(data.shape[0] == labels.shape[0])
    #7400 --> 10 x 740 7400, 1024 x 12
    data = data[0:7400]
    labels = labels[0:7400]
    batched_data = tf.reshape(data, (folds*number_of_batches,-1,1024,12))
    batched_labels = tf.reshape(labels, (folds*number_of_batches,-1))

    return batched_data, batched_labels




def train(model, train_inputs, train_labels):
    # squeezed_inputs = tf.squeeze(train_inputs,axis=0)
    # squeezed_labels = tf.squeeze(train_labels,axis=0)
    with tf.GradientTape() as tape:
        probs = model.call(train_inputs)
        loss = model.loss_function(probs,train_labels)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients,model.trainable_variables))




def test(model, test_inputs, test_labels):
    # squeezed_inputs = tf.squeeze(test_inputs,axis=0)
    # squeezed_labels = tf.squeeze(test_labels,axis=0)
    probs = model.call(test_inputs)
    accuracy = model.accuracy_function(probs,test_labels)
    return accuracy

def main():
    data, labels = preprocess.get_data("./MillionSongSubset/")
    model = Inception_Model()
    num_epochs = 5
    j=0
    folds = 10
    number_of_batches = 10
    while j <= num_epochs:
        num_songs = data.shape[0]
        idx = np.arange(num_songs)
        shuffled = tf.random.shuffle(idx)
        shuffled_inputs = tf.gather(data, shuffled, axis=0)
        shuffled_labels = tf.gather(labels, shuffled, axis=0)
        batched_data, batched_labels = create_batched_kfolds(shuffled_inputs, shuffled_labels, folds,number_of_batches)
        epoch_acc = []
        for i in range(0, int(batched_data.shape[0]/number_of_batches)):
            start_index = (i*number_of_batches)
            end_index = ((i+1)*number_of_batches)
            if i == 0:
                train_data = batched_data[end_index:batched_data.shape[0]]
                train_data = tf.expand_dims(train_data,-1)
                train_labels = batched_labels[end_index:batched_data.shape[0]]
                train_labels = tf.expand_dims(train_labels,-1)
                test_data = batched_data[start_index:end_index]
                test_data = tf.expand_dims(test_data,-1)
                test_labels = batched_labels[start_index:end_index]
                test_labels = tf.expand_dims(test_labels,-1)
            else:
                train_data = tf.concat([batched_data[:start_index],batched_data[end_index:]],axis=0)
                train_data = tf.expand_dims(train_data,-1)
                train_labels = tf.concat([batched_labels[:start_index],batched_labels[end_index:]],axis=0)
                train_labels = tf.expand_dims(train_labels,-1)
                test_data = batched_data[start_index:end_index]
                test_data = tf.expand_dims(test_data,-1)
                test_labels = batched_labels[start_index:end_index]
                test_labels = tf.expand_dims(test_labels,-1)

            # print("Train Data and Label Shape:", tf.shape(train_data),tf.shape(train_labels))
            # print("Test Data and Label Shape:", tf.shape(test_data),tf.shape(test_labels))
            train_data = tf.cast(train_data,dtype=tf.float32)
            test_data = tf.cast(test_data,dtype=tf.float32)

            for train_batch in range(tf.shape(train_data)[0]):
                train(model,train_data[train_batch],train_labels[train_batch])
            batch_acc = []
            for test_batch in range(tf.shape(test_data)[0]):
                average_batch_accuracy = test(model,test_data[test_batch],test_labels[test_batch])
                batch_acc.append(average_batch_accuracy)
            fold_acc = tf.reduce_mean(batch_acc)
            epoch_acc.append(fold_acc)
            print("Epoch ", j, "Fold", i, " Average Accuracy: ", fold_acc)
        print("Epoch ",j," Average Accuracy: ", tf.reduce_mean(epoch_acc))

        j+=1



if __name__ == '__main__':
    main()
