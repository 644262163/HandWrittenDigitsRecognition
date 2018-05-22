# coding=utf-8

import os
import shutil
import tensorflow as tf
import input_data

# 创建tensorflow的默认会话
sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 获取数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# 卷积层1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 激活函数 计算修正线性单元 将矩阵中每行的非最大值置0
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# 池化层1
# 最大值池化
h_pool1 = max_pool_2x2(h_conv1)

# 卷积层2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# 激活函数 计算修正线性单元 将矩阵中每行的非最大值置0
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 池化层2
# 最大值池化
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层1
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# 激活函数 计算修正线性单元 将矩阵中每行的非最大值置0
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
# Dropout 随机删除网络中的一些隐藏神经元 保持输入输出神经元不变 可以比较有效地减轻过拟合的发生
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接层2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# 每个值表示这个样本属于每个类的概率
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

# 自适应矩估计
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 正确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 初始化
sess.run(tf.global_variables_initializer())

# 训练
for i in range(20000):
    batch = mnist.train.next_batch(50)

    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g" % (i, train_accuracy)

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 保存权重
dir_path = os.path.join(os.getcwd(), 'model')
if os.path.exists(dir_path):
    shutil.rmtree(dir_path)
os.mkdir(dir_path)
saver = tf.train.Saver()
save_path = saver.save(sess, "./model/model.ckpt")
print "Model saved in file: %s" % (save_path)