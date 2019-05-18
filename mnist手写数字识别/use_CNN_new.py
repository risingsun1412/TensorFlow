import tensorflow as tf
from os import environ
from tensorflow.examples.tutorials.mnist import input_data

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_w(shape):
    return tf.Variable(tf.truncated_normal(shape = shape, stddev = 0.1))
def get_b(shape):
    return tf.Variable(tf.constant(0.1, shape = shape))
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1,1,1,1], padding = "SAME")
def max_pool(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")
def Main():
    x = tf.placeholder(tf.float32, shape = [None, 28*28])
    y_true = tf.placeholder(tf.float32, shape = [None, 10])

    conv1_w = get_w([5,5,1,32])
    conv1_b = get_b([32])
    x_reshape = tf.reshape(x, [-1,28,28,1])
    conv1_conv = conv2d(x_reshape, conv1_w)
    conv1_relu = tf.nn.relu(conv1_conv + conv1_b)
    conv1_out = max_pool(conv1_relu)

    conv2_w = get_w([5,5,32,64])
    conv2_b = get_b([64])
    conv2_conv = conv2d(conv1_out, conv2_w)
    conv2_relu = tf.nn.relu(conv2_conv + conv2_b)
    conv2_out = max_pool(conv2_relu)

    fc1_w = get_w([7*7*64,512])
    fc1_b = get_b([512])
    conv2_out_reshape = tf.reshape(conv2_out, [-1,7*7*64])
    fc1_matmul = tf.matmul(conv2_out_reshape, fc1_w)
    fc1_relu = tf.nn.relu(fc1_matmul + fc1_b)

    fc2_w = get_w([512,1024])
    fc2_b = get_b([1024])
    fc2_matmul = tf.matmul(fc1_relu, fc2_w)
    fc2_relu = tf.nn.relu(fc2_matmul + fc2_b)

    sm_w = get_w([1024,10])
    sm_b = get_b([10])
    sm_matmul = tf.matmul(fc2_relu, sm_w)
    y_predict = tf.nn.softmax(sm_matmul + sm_b)

    loss = tf.reduce_mean(-tf.reduce_sum(y_true*tf.log(y_predict)))
    opt = tf.train.AdamOptimizer(1e-4)
    train = opt.minimize(loss)
    correct_prediction = tf.equal(tf.math.argmax(y_predict, 1), tf.math.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        mnist = input_data.read_data_sets('C:/Users/rising_sun/Desktop/neural_network/MNIST_data/', one_hot=True)
        for i in range(3000):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                print('第%d轮训练后,正确率为%.2f %%'%(i, 100*sess.run(accuracy, feed_dict = {x: batch[0], y_true: batch[1]})))
            sess.run(train, feed_dict = {x: batch[0], y_true: batch[1]})

        print('最终在测试集上的正确率为%.2f %%'%(100*sess.run(accuracy, feed_dict = {x: mnist.test.images, y_true: mnist.test.labels})))
        

if __name__ == '__main__':
    Main()
