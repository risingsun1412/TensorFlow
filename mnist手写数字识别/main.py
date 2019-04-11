import numpy as np
import tensorflow as tf
from os import environ
from tensorflow.examples.tutorials.mnist import input_data
#environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #关闭tensorflow输出硬件信息

def start_to_train():
    mnist = input_data.read_data_sets('./test_data/', one_hot = True)#mnist数据集
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))#权值
    bias = tf.Variable(tf.zeros([10]))#bias,偏置量
    y = tf.nn.softmax(tf.matmul(x, w) + bias)#tf自带的nn里面的softmax模型
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))#使用交叉熵作为损失函数(loss)
    opt = tf.train.GradientDescentOptimizer(0.5)
    train = opt.minimize(cross_entropy)
    init = tf.global_variables_initializer()#初始化所有变量
    with tf.Session() as sess:#python上下文管理器
        sess.run(init)
        for i in range(3000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train, feed_dict = {x: batch_xs, y_: batch_ys})#使用feed_dict = {}喂数据
        
        correction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(y_, 1))#旧版时tf.argmax()
        accuracy = tf.reduce_mean(tf.cast(correction, tf.float32))#计算正确(1),错误(0),的平均值
        print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == "__main__":
    start_to_train()
