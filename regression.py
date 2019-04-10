import os
import tensorflow as tf
import input_data
import model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 训练数据，自动下载，结构如下
# xs [[0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]*28] 28*28
# ys [0,0,1,0,0,0,0,0,0,0] 1*10 例数据表示2
data = input_data.read_data_sets('data', one_hot=True)

with tf.variable_scope("regression"):
    # 定义输入数据参数
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
    # 模型
    y, variables = model.regression(xs)

# 计算偏差
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# 与预测值进行比较
# argmax x=1时返回每行中最大值的索引，识别的结果集y如果下标相等则认为一样
correct_prediction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 保存训练得到的模型
saver = tf.train.Saver(variables)

# 开始训练
with tf.Session() as sess:
    # 初使化全局参数
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        # 获取100个训练数据
        batch_xs, batch_ys = data.train.next_batch(100)
        if i % 100 == 0:
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
            # 打印出结果
            result = sess.run(accuracy, feed_dict={xs: data.test.images, ys: data.test.labels})
            print(i, result)

    path = saver.save(
        sess,
        os.path.join(os.path.dirname(__file__), 'models', 'regression.ckpt'),
        write_meta_graph=False,
        write_state=False
    )
    print("Saved:", path)
