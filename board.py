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

def variable_summaries(var):
    with tf.name_scope('summaries'):
        # 计算参数的均值
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # 计算参数的标准差 最大值 最小值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)

# 获取数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.name_scope('input'):
    x = tf.placeholder("float", [None, 784])
    y_ = tf.placeholder("float", [None, 10])

with tf.name_scope('input_reshape'):
    # 图片数量, 图片高度, 图片宽度, 图像通道数
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 10)

# 卷积层1
with tf.name_scope('conv1'):
    # 调用之前的方法初始化权重w 并且调用参数信息的记录方法 记录w的信息
    with tf.name_scope('weights'):
        # filter_height, filter_width, in_channels, out_channels
        W_conv1 = weight_variable([5, 5, 1, 32])
        variable_summaries(W_conv1)
    # 调用之前的方法初始化权重b 并且调用参数信息的记录方法 记录b的信息
    with tf.name_scope('biases'):
        b_conv1 = bias_variable([32])
        variable_summaries(b_conv1)
    # 执行wx+b的线性计算，并且用直方图记录下来
    with tf.name_scope('linear_compute'):
        preactivate = conv2d(x_image, W_conv1) + b_conv1
        tf.summary.histogram('linear', preactivate)
    # 激活函数 计算修正线性单元 将矩阵中每行的非最大值置0
    h_conv1 = tf.nn.relu(preactivate)
    # 将线性输出经过激励函数 并将输出也用直方图记录下来
    tf.summary.histogram('activations', h_conv1)

# 池化层1
with tf.name_scope('pool1'):
    # 最大值池化
    h_pool1 = max_pool_2x2(h_conv1)

# 卷积层2
with tf.name_scope('conv2'):
    # 调用之前的方法初始化权重w 并且调用参数信息的记录方法 记录w的信息
    with tf.name_scope('weights'):
        # filter_height, filter_width, in_channels, out_channels
        W_conv2 = weight_variable([5, 5, 32, 64])
        variable_summaries(W_conv2)
    # 调用之前的方法初始化权重b 并且调用参数信息的记录方法 记录b的信息
    with tf.name_scope('biases'):
        b_conv2 = bias_variable([64])
        variable_summaries(b_conv2)
    # 执行wx+b的线性计算，并且用直方图记录下来
    with tf.name_scope('linear_compute'):
        preactivate = conv2d(h_pool1, W_conv2) + b_conv2
        tf.summary.histogram('linear', preactivate)
    # 激活函数 计算修正线性单元 将矩阵中每行的非最大值置0
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # 将线性输出经过激励函数 并将输出也用直方图记录下来
    tf.summary.histogram('activations', h_conv2)

# 池化层2
with tf.name_scope('pool2'):
    # 最大值池化
    h_pool2 = max_pool_2x2(h_conv2)

# 全连接层1
with tf.name_scope('fc1'):
    # 调用之前的方法初始化权重w 并且调用参数信息的记录方法 记录w的信息
    with tf.name_scope('weights'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        variable_summaries(W_fc1)
    # 调用之前的方法初始化权重b 并且调用参数信息的记录方法 记录b的信息
    with tf.name_scope('biases'):
        b_fc1 = bias_variable([1024])
        variable_summaries(b_fc1)
    # 执行wx+b的线性计算，并且用直方图记录下来
    with tf.name_scope('linear_compute'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        preactivate = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        tf.summary.histogram('linear', preactivate)
    # 激活函数 计算修正线性单元 将矩阵中每行的非最大值置0
    h_fc1 = tf.nn.relu(preactivate)
    # 将线性输出经过激励函数 并将输出也用直方图记录下来
    tf.summary.histogram('activations', h_fc1)

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder("float")
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    # Dropout 随机删除网络中的一些隐藏神经元 保持输入输出神经元不变 可以比较有效地减轻过拟合的发生
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接层2
with tf.name_scope('fc2'):
    # 调用之前的方法初始化权重w 并且调用参数信息的记录方法 记录w的信息
    with tf.name_scope('weights'):
        W_fc2 = weight_variable([1024, 10])
        variable_summaries(W_fc2)
    # 调用之前的方法初始化权重b 并且调用参数信息的记录方法 记录b的信息
    with tf.name_scope('biases'):
        b_fc2 = bias_variable([10])
        variable_summaries(b_fc2)
    # 执行wx+b的线性计算，并且用直方图记录下来
    with tf.name_scope('linear_compute'):
        preactivate = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        tf.summary.histogram('linear', preactivate)
    # 每个值表示这个样本属于每个类的概率
    y_conv = tf.nn.softmax(preactivate)
    # 将线性输出经过激励函数 并将输出也用直方图记录下来
    tf.summary.histogram('activations', y_conv)

with tf.name_scope('train'):
    # 交叉熵
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    tf.summary.scalar('loss', cross_entropy)
    # 自适应矩估计
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('assess'):
    with tf.name_scope('correct_prediction'):
        # 正确率
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy', accuracy)

# 初始化
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
# 写到指定的磁盘路径中
train_writer = tf.summary.FileWriter('./log/train', sess.graph)
test_writer = tf.summary.FileWriter('./log/test')

# 训练
for i in range(20000):
    batch = mnist.train.next_batch(50)

    if i % 100 == 0:
        summary_str = sess.run(merged, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        test_writer.add_summary(summary_str, i)

        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g" % (i, train_accuracy)

    summary_str, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    train_writer.add_summary(summary_str, i)

train_writer.close()
test_writer.close()