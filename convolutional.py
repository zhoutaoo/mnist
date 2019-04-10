import os
import tensorflow as tf
import input_data
import model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 训练数据，自动下载，结构如下
# xs [[0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]*28] 28*28
# ys [0,0,1,0,0,0,0,0,0,0] 1*10 例数据表示2
data = input_data.read_data_sets('data', one_hot=True)

# 定义模型
with tf.variable_scope('convolutional'):
    xs = tf.placeholder(tf.float32, [None, 784], name='x')
    ys = tf.placeholder('float', [None, 10], name='y')
    # 模型
    keep_prob = tf.placeholder(tf.float32)
    y, variables = model.convolutional(xs, keep_prob)

# 计算偏差
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(y)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 与预测值进行比较
# argmax x=1时返回每行中最大值的索引，识别的结果集y如果下标相等则认为一样
correct_prediction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 保存训练得到的模型
saver = tf.train.Saver(variables)

# 开始训练
with tf.Session() as sess:
    merged_summary_op = tf.summary.merge_all()
    summary_write = tf.summary.FileWriter('view/', sess.graph)
    summary_write.add_graph(sess.graph)
    # 初使化变量
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        # 获取100个训练数据
        batch_xs, batch_ys = data.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1.0})
            print("Step %d, training accuracy %g" % (i, train_accuracy))
        # 训练
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})

    # 使用测试集验证结果
    print(sess.run(accuracy, feed_dict={xs: data.test.images, ys: data.test.labels, keep_prob: 1.0}))

    path = saver.save(
        sess,
        os.path.join(os.path.dirname(__file__), 'models', 'convolutional.ckpt'),
        write_meta_graph=False,
        write_state=False
    )
    print("Saved:", path)
