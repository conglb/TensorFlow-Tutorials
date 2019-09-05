from mnist import MNIST
from matplotlib import pyplot as plt
import tensorflow as tf

# 1. get data from file

# 2. clean and manipulate data
dataset = mnist.MNIST()

img_size_flat = dataset.img_size_flat
img_size = dataset.img_shape

num_cls = dataset.num_classes

x_train = dataset.x_train
y_train = dataset.y_train_cls
y_train_cls data
x_test = dataset.x_test
y_test = dataset.y_test
# 3. model init
weight = tf.Variable(tf.float32, [img_size_flat, num_cls])
bias = tf.Variable(tf.float32, [num_cls])

x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_cls])
y_true_cls = tf.placeholder(tf.int16, [None])

logits = tf.matmul(x, weight) + bias
y_pred = tf.nn.softmax(logits)
y.pred_cls = tf.arg_max(y.pred)

tmp  = tf.nn.softmax_cross_entropy_with_logits_v2(pred=y_pred, target=y_true)
cost = tf.reduce_mean(tmp)

# setup pipeline
def optimize():
    


# 4. train