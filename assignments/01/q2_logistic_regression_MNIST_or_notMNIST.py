""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
import sys
import tensorflow as tf
import time

import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

# Step 1: Read in data
if sys.argv[1] == 'mnist':
    mnist_folder = 'data/{0}'.format(sys.argv[1])
    utils.download_mnist(mnist_folder)
elif sys.argv[1] == 'not_mnist':
    mnist_folder = 'data/{0}'.format(sys.argv[1])
    # download data manually from https://github.com/davidflanagan/notMNIST-to-MNIST
else:
    raise ValueError('You would need to download and format data for {0} '
                     'the MNIST way'.format(sys.argv[1]))
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
print('Create training Dataset and batch it...')
# NOTE: it appears to be much faster if this step is done in GPU
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)  # if you want to shuffle your data
train_data = train_data.batch(batch_size)

print('create testing Dataset and batch it...')
test_data = tf.data.Dataset.from_tensor_slices(test)
# not point to shuffle
test_data = test_data.batch(batch_size)

# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(
    train_data.output_types,
    train_data.output_shapes
)

img, label = iterator.get_next()

# initializer for train_data
train_init = iterator.make_initializer(train_data)
# initializer for test data
test_init = iterator.make_initializer(test_data)

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
num_classes = 10                # MNIST: 0 to 9
w = tf.get_variable(
    name='weights',
    shape=(img.shape[1].value, num_classes),  # shape[0] is the batch_size
    initializer=tf.random_normal_initializer(0, 0.01))
b = tf.get_variable(
    name='bias',
    shape=(1, num_classes),
    initializer=tf.zeros_initializer())

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
logits = tf.matmul(img, w) + b

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=label, logits=logits, name="entropy")
loss = tf.reduce_mean(entropy, name="loss")

# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:

    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs):
        # drawing samples from train_data
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)			# drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/n_test))
writer.close()
