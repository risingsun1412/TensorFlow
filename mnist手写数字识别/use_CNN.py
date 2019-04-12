import numpy as np
import tensorflow as tf
from os import environ, path
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import matplotlib.pyplot as plt

environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 关闭tensorflow输出硬件信息

def get_test_data():
    img = Image.open('C:/Users/rising_sun/Pictures/8.png')# 自己制作的28*28像素的数字图片位置
    img = img.convert('L')
    tv = list(img.getdata()) 
    tva = [(255-x)*1.0/255.0 for x in tv] 
    return tva, img
def w_var(shape):
    w = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(w)
def b_var(shape):
    b = tf.constant(0.1, shape = shape)
    return tf.Variable(b)
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
def train():
    #定义place_holder
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    # 第一层卷积
    w_conv1 = w_var([5, 5, 1, 32])
    b_conv1 = b_var([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积
    w_conv2 = w_var([5, 5, 32, 64])
    b_conv2 = b_var([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层
    w_fc1 = w_var([7*7*64, 1024])
    b_fc1 = b_var([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # Dropout减少过拟合
    keep_prob = tf.placeholder('float')# keep_prob: 每个数据被保留的概率
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 输出层
    w_fc2 = w_var([1024, 10])
    b_fc2 = b_var([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    # 准备启动
    # loss = 交叉熵 + L2正则
    loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv))) + tf.contrib.layers.l2_regularizer(0.5)(w_conv1) + tf.contrib.layers.l2_regularizer(0.5)(w_conv2) + tf.contrib.layers.l2_regularizer(0.5)(w_fc1) + tf.contrib.layers.l2_regularizer(0.5)(w_fc2)
    opt = tf.train.AdamOptimizer(1e-4)
    train = opt.minimize(loss)
    correct_prediction = tf.equal(tf.math.argmax(y_conv, 1), tf.math.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        if not path.exists('c:/Users/rising_sun/Desktop/neural_network/models'):# 模型不存在, 开始训练
            print('Model not exist!!!')
            mnist = input_data.read_data_sets('c:/Users/rising_sun/Desktop/neural_network/MNIST_data/', one_hot=True)# 打开数据集准备训练
            for i in range(30000):
                batch = mnist.train.next_batch(100)
                if i % 100 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0})
                    print('After %d steps, training accuracy is %g %%'%(i, train_accuracy * 100))
                sess.run(train, feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
            saver.save(sess, r'c:/Users/rising_sun/Desktop/neural_network/models/model.ckpt')
        else:# 模型存在, 直接恢复模型
            print('Model exist!!!')
            saver.restore(sess, 'c:/Users/rising_sun/Desktop/neural_network/models/model.ckpt')
        
        prediction = tf.math.argmax(y_conv, 1)
        testdt, img = get_test_data()
        res_int = sess.run(prediction, feed_dict = {x: [testdt], keep_prob: 1.0})
        print('The prediction of number is:', res_int)
        plt.imshow(img)
        plt.show()
        # mnist = input_data.read_data_sets('c:/Users/rising_sun/Desktop/neural_network/MNIST_data/', one_hot=True)
        # print('test accuracy on mnist is %g %%'%(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) * 100))# 最终模型在mnist数据集上的正确率
if __name__ == "__main__":
    train()
