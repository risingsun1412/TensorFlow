import numpy as np
import os
import tensorflow as tf
#import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #关闭tensorflow输出硬件信息

def generate_data():#随机生成测试数据
    num_points = 1000
    vector_set = []
    for i in range(num_points):
        x1 = np.random.normal(0.0, 0.55)
        y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)#以函数y = 0.1x+0.3为基准生成点数据
        vector_set.append([x1, y1])
        x_data = [v[0] for v in vector_set]#就是vector_set里面的所有x1组成的列表
        y_data = [v[1] for v in vector_set]#同上
        #plt.scatter(x_data, y_data, c = 'r')
    #plt.show()
    return x_data, y_data

def train(x_data, y_data):
    w = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name = 'w')#生成均匀分布的值,其中[1]可以换成(1, ),表示矩阵的形状
    b = tf.Variable(tf.zeros([1]), name = 'b')#b初始化为0
    y = w * x_data + b#根据随机生成的w, x_data, b计算y
    loss = tf.reduce_mean(tf.square(y - y_data), name = 'loss')#tf.square()平方,tf.reduce_mean(不指定axis的情况下)就是计算平均值,所以loss就是标准差
    optimizer = tf.train.GradientDescentOptimizer(0.5)#设置学习率为0.5
    train = optimizer.minimize(loss, name = 'train')#使用优化器通过损失函数调整神经网络权值

    sess = tf.Session()#开启任务，为了方便,起了别名sess
    init = tf.global_variables_initializer()#同上
    sess.run(init)#初始化全部变量

    print('w = ', sess.run(w), 'b = ', sess.run(b), 'loss = ', sess.run(loss))#这是随机生成的w和b
    for step in range(50):#一共训练50次
        sess.run(train)
        print('w = ', sess.run(w), 'b = ', sess.run(b), 'loss = ', sess.run(loss))#这是训练后的w和b
    
if __name__ == "__main__":
    x_data, y_data = generate_data()
    train(x_data, y_data)
