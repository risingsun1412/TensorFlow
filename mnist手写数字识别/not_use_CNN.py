import numpy as np
import tensorflow as tf
from os import environ, path
from tensorflow.examples.tutorials.mnist import input_data

environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 关闭tensorflow输出硬件信息

def save_model(sess, file):
    tf.train.Saver().save(sess, file)
def restore_model(sess, file):
    tf.train.Saver().restore(sess, file)
def get_test_pic():
    img = Image.open('C:/Users/rising_sun/Pictures/9.png')# 自己制作的28*28像素的数字图片位置
    # plt.imshow(img)
    # plt.show()
    img = img.convert('L')
    tv = list(img.getdata()) 
    tva = [(255-x)*1.0/255.0 for x in tv] 
    return tva
def start_to_train(pic_to_test):
    mnist = input_data.read_data_sets('./test_data/', one_hot = True)# mnist数据集
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))# 权值
    bias = tf.Variable(tf.zeros([10]))# bias,偏置量
    y = tf.nn.softmax(tf.matmul(x, w) + bias)# tf自带的nn里面的softmax模型,计算预测的数字
    y_ = tf.placeholder(tf.float32, [None, 10])# 数据集里的正确数字
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))# 使用交叉熵作为损失函数(loss)
    opt = tf.train.GradientDescentOptimizer(0.5)
    train = opt.minimize(cross_entropy)
    init = tf.global_variables_initializer()# 初始化所有变量
    
    with tf.Session() as sess:# python上下文管理器
        sess.run(init)
        file = 'C:/Users/rising_sun/Desktop/neural_network/models/'# 模型保存位置
        model_name = 'model.ckpt'# 模型名称
        if path.exists(file):# 检查模型是否存在
            restore_model(sess, file + model_name)# 恢复模型
            print('models exist!!!')
        else:
            print('models not exists!!!')
            for i in range(3000):# 迭代3000次
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train, feed_dict = {x: batch_xs, y_: batch_ys})# 使用feed_dict = {}喂数据
            save_model(sess, file + model_name)# 存储模型

        prediction = tf.math.argmax(y, 1)
        res_int = sess.run(prediction, feed_dict = {x:[pic_to_test]})
        print('The prediction of number is:',res_int)

        # 在mnist数据集提供的测试数据上测试模型正确率
        # correction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(y_, 1))# 旧版是tf.argmax()
        # accuracy = tf.reduce_mean(tf.cast(correction, tf.float32))# 计算正确(1),错误(0),的平均值
        # print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == "__main__":
    pic = get_test_pic()
    start_to_train(pic)
