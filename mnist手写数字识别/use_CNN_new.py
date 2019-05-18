import numpy as np
import tensorflow as tf
from os import environ, path
from tensorflow.examples.tutorials.mnist import input_data

environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 关闭tensorflow输出硬件信息

def get_w(shape, name):
    w = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(w, name = name)

def get_b(shape, name):
    b = tf.constant(0.1, shape = shape)
    return tf.Variable(b, name = name)
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def Main():
    x = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.float32, [None, 10])
    
    # 第一层卷积
    conv1_w = get_w([5, 5, 1, 32], 'conv1_w')
    conv1_b = get_b([32], 'conv1_b')
    x_ = tf.reshape(x, [-1, 28, 28, 1])
    conv1_relu = tf.nn.relu(conv2d(x_, conv1_w) + conv1_b)
    conv1_pool = max_pool_2x2(conv1_relu)

    # 第二层卷积
    conv2_w = get_w([5, 5, 32, 64], 'conv2_w')
    conv2_b = get_b([64], 'conv2_b')
    conv2_relu = tf.nn.relu(conv2d(conv1_pool, conv2_w) + conv2_b)
    conv2_pool = max_pool_2x2(conv2_relu)

    # 全连接层
    fc1_w = get_w([7*7*64, 512], 'fc1_w')
    fc1_b = get_b([512], 'fc1_b')
    conv2_pool_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
    fc1_relu = tf.nn.relu(tf.matmul(conv2_pool_flat, fc1_w) + fc1_b)

    # Dropout减少过拟合
    keep_prob = tf.placeholder('float')# keep_prob: 每个神经元不工作的概率
    fc1_dropout = tf.nn.dropout(fc1_relu, keep_prob)

    # softmax层
    fc2_w = get_w([512, 10], 'fc2_w')
    fc2_b = get_b([10], 'fc2_b')
    y_predict = tf.nn.softmax(tf.matmul(fc1_dropout, fc2_w) + fc2_b)

    # loss = 交叉熵
    loss = tf.reduce_mean(-tf.reduce_sum(y_true*tf.log(y_predict)))
    opt = tf.train.AdamOptimizer(1e-4)
    train = opt.minimize(loss)
    correct_prediction = tf.equal(tf.math.argmax(y_predict, 1), tf.math.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        mnist = input_data.read_data_sets('C:/Users/rising_sun/Desktop/neural_network/MNIST_data/', one_hot=True)
        for i in range(10000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict = {x:batch[0], y_true:batch[1], keep_prob:1.0})
                print('After %d steps, training accuracy is %.2f %%'%(i, train_accuracy * 100))
            sess.run(train, feed_dict = {x: batch[0], y_true: batch[1], keep_prob: 0.5})
            
        print('test accuracy on mnist is %.2f %%'%(sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels, keep_prob: 1.0}) * 100))# 最终模型在mnist数据集上的正确率
        
if __name__ == "__main__":
    Main()
